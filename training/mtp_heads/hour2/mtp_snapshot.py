from __future__ import annotations

import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

INDEX = "snapshot.json"
LM_HEAD = "lm_head.pt"
EMBED = "embed.pt"
DR_PART = "dr"
DR_FIELDS = [("id", "<i4"), ("conf", "<f2"), ("acc", "u1")]
DR_NONE = 255
CGROUP = Path("/sys/fs/cgroup")
GiB = 2**30


def quantize_rows(multi: torch.Tensor, hc: int) -> tuple[torch.Tensor, torch.Tensor]:
    rows = multi.shape[0]
    view = multi.float().reshape(rows, hc, -1)
    scale = (view.abs().amax(-1) / 127.0).clamp(min=1e-8)
    q = (view / scale[..., None]).round().clamp(-127, 127).to(torch.int8)
    return q.reshape(rows, -1), scale.to(torch.float16)


def dequantize_rows(q: torch.Tensor, scale: torch.Tensor, hc: int,
                    dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    rows = q.shape[0]
    view = q.float().reshape(rows, hc, -1) * scale.float()[..., None]
    return view.reshape(rows, -1).to(dtype)


GROUP4 = 128


def group_for(width: int, hc: int) -> int:
    return math.gcd(width // hc, GROUP4)


CHUNK4 = 4096


def quantize_rows4(multi: torch.Tensor, group: int = GROUP4) -> tuple[torch.Tensor, torch.Tensor]:
    if multi.shape[0] > CHUNK4:
        parts = [quantize_rows4(multi[at:at + CHUNK4], group) for at in range(0, multi.shape[0], CHUNK4)]
        return torch.cat([one[0] for one in parts]), torch.cat([one[1] for one in parts])
    rows = multi.shape[0]
    view = multi.float().reshape(rows, -1, group)
    scale = (view.abs().amax(-1) / 7.0).clamp(min=1e-8)
    q = (view / scale[..., None]).round().clamp(-7, 7).to(torch.int16) + 8
    flat = q.reshape(rows, -1)
    packed = (flat[:, 0::2] | (flat[:, 1::2] << 4)).to(torch.uint8)
    return packed, scale.reshape(rows, -1).to(torch.float16)


def dequantize_rows4(packed: torch.Tensor, scale: torch.Tensor, group: int = GROUP4,
                     dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    if packed.shape[0] > CHUNK4:
        return torch.cat([dequantize_rows4(packed[at:at + CHUNK4], scale[at:at + CHUNK4], group, dtype)
                          for at in range(0, packed.shape[0], CHUNK4)])
    rows = packed.shape[0]
    wide = packed.to(torch.int16)
    flat = torch.stack(((wide & 15) - 8, (wide >> 4) - 8), -1).reshape(rows, -1).float()
    view = flat.reshape(rows, -1, group) * scale.float().reshape(rows, -1)[..., None]
    return view.reshape(rows, -1).to(dtype)


def stored_rows(multi: torch.Tensor, hc: int, bits: int) -> torch.Tensor:
    if bits == 4:
        group = group_for(multi.shape[1], hc)
        return dequantize_rows4(*quantize_rows4(multi, group), group, multi.dtype)
    return dequantize_rows(*quantize_rows(multi, hc), hc, multi.dtype)


def row_bytes(width: int, hc: int, topk: int, bits: int = 8) -> int:
    hidden = width + 2 * hc if bits == 8 else width // 2 + 2 * (width // group_for(width, hc))
    return hidden + 6 * topk + 4


def hidden_shapes(width: int, hc: int, bits: int) -> dict[str, tuple[Any, int]]:
    if bits == 4:
        return {"q": (np.uint8, width // 2), "s": (np.float16, width // group_for(width, hc))}
    return {"q": (np.int8, width), "s": (np.float16, hc)}


def cgroup_held_gib(base: Path = CGROUP) -> tuple[float | None, float | None]:
    try:
        stat = dict(line.split() for line in (base / "memory.stat").read_text(encoding="utf-8").splitlines()
                    if len(line.split()) == 2)
        held = (int(stat.get("anon", 0)) + int(stat.get("shmem", 0))) / GiB
        ceiling = (base / "memory.max").read_text(encoding="utf-8").strip()
        return held, (int(ceiling) / GiB if ceiling.isdigit() else None)
    except (OSError, ValueError):
        return None, None


class Writer:
    def __init__(self, folder: Path, hc: int, width: int, topk: int, train_bits: int = 8):
        if train_bits not in (4, 8):
            raise ValueError(f"train rows are stored in 8 or 4 bits, not {train_bits}")
        self.folder = folder
        self.hc = hc
        self.width = width
        self.topk = topk
        self.train_bits = train_bits
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)
        self.segments: list[dict[str, Any]] = []
        self.entries: list[dict[str, Any]] = []
        self.current: list[int] = []
        self.bytes = 0
        self.extended = 0

    def row_bytes(self, bits: int = 8) -> int:
        return row_bytes(self.width, self.hc, self.topk, bits)

    def files(self, seg: int) -> dict[str, Path]:
        return {part: self.folder / f"seg{seg:05d}.{part}" for part in ("ids", "q", "s", "tp", "ti")}

    def append(self, seg: int, ids: list[int], multi: torch.Tensor, probs: torch.Tensor, tids: torch.Tensor) -> None:
        bits = int(self.segments[seg].get("bits", 8))
        q, scale = quantize_rows4(multi, group_for(self.width, self.hc)) if bits == 4 else quantize_rows(multi, self.hc)
        parts = {"ids": np.asarray(ids, dtype=np.int32), "q": q.cpu().numpy(), "s": scale.cpu().numpy(),
                 "tp": probs.to(torch.float16).cpu().numpy(), "ti": tids.to(torch.int32).cpu().numpy()}
        for part, path in self.files(seg).items():
            with open(path, "ab") as handle:
                handle.write(np.ascontiguousarray(parts[part]).tobytes())
        self.bytes += len(ids) * self.row_bytes(bits)

    def add(self, record: dict[str, Any], ids: list[int], start: int, new_rows, targets) -> dict[str, Any]:
        bits = self.train_bits if record["split"] == "train" else 8
        reuse = bool(self.segments) and int(self.segments[-1].get("bits", 8)) == bits \
            and len(ids) >= len(self.current) and ids[:len(self.current)] == self.current
        if reuse:
            seg = len(self.segments) - 1
            first = len(self.current)
            self.extended += 1
        else:
            seg = len(self.segments)
            self.segments.append({"rows": 0, "bits": bits})
            self.current = []
            first = 0
        if len(ids) > first:
            multi = new_rows(first)
            probs, tids = targets(first)
            self.append(seg, ids[first:], multi, probs, tids)
            self.segments[seg]["rows"] = len(ids)
            self.current = list(ids)
        entry = {"key": str(record.get("id") or record.get("stem")), "split": record["split"],
                 "game": str(record.get("game")), "run": str(record.get("run")), "step": int(record.get("step") or 0),
                 "call": int(record.get("call") or 0), "seg": seg, "count": len(ids), "start": int(start)}
        self.entries.append(entry)
        return entry

    def close(self, meta: dict[str, Any]) -> dict[str, Any]:
        index = {"hc": self.hc, "width": self.width, "topk": self.topk, "segments": self.segments,
                 "entries": self.entries, "bytes": self.bytes, **meta}
        (self.folder / INDEX).write_text(json.dumps(index), encoding="utf-8")
        return index


PARTS = ("ids", "q", "s", "tp", "ti")


def save_subset(source: Path, target: Path, budget: int, extra: list[Path]) -> dict[str, Any]:
    index = json.loads((source / INDEX).read_text(encoding="utf-8"))

    def size_of(seg: int) -> int:
        bits = int(index["segments"][seg].get("bits", 8))
        return int(index["segments"][seg]["rows"]) * row_bytes(int(index["width"]), int(index["hc"]),
                                                                int(index["topk"]), bits)

    held = sorted({one["seg"] for one in index["entries"] if one["split"] == "heldout"})
    rest = [seg for seg in range(len(index["segments"])) if seg not in set(held)]
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    spent = sum(path.stat().st_size for path in extra if path.is_file())
    kept: list[int] = []
    for seg in [*held, *rest]:
        size = size_of(seg)
        if seg not in held and spent + size > budget:
            continue
        kept.append(seg)
        spent += size
    remap = {seg: at for at, seg in enumerate(kept)}
    for seg in kept:
        for part in PARTS:
            shutil.copyfile(source / f"seg{seg:05d}.{part}", target / f"seg{remap[seg]:05d}.{part}")
    for path in extra:
        if path.is_file():
            shutil.copyfile(path, target / path.name)
    entries = [{**one, "seg": remap[one["seg"]]} for one in index["entries"] if one["seg"] in remap]
    saved = {**index, "segments": [index["segments"][seg] for seg in kept], "entries": entries,
             "bytes": sum(size_of(seg) for seg in kept), "saved_from": str(source)}
    (target / INDEX).write_text(json.dumps(saved), encoding="utf-8")
    return {"segments": len(kept), "of": len(index["segments"]), "bytes": spent,
            "heldout": sum(one["split"] == "heldout" for one in entries),
            "train": sum(one["split"] == "train" for one in entries)}


class Reader:
    def __init__(self, folder: Path, device: torch.device, window: int | None = None):
        self.folder = folder
        self.device = device
        self.window = window
        self.index = json.loads((folder / INDEX).read_text(encoding="utf-8"))
        self.hc = int(self.index["hc"])
        self.width = int(self.index["width"])
        self.topk = int(self.index["topk"])
        self.maps: dict[int, dict[str, np.memmap]] = {}
        self.reads = 0
        self.seconds = 0.0

    def entries(self, split: str) -> list[dict[str, Any]]:
        return [one for one in self.index["entries"] if one["split"] == split]

    def mapped(self, seg: int) -> dict[str, np.memmap]:
        if seg not in self.maps:
            rows = int(self.index["segments"][seg]["rows"])
            hidden = hidden_shapes(self.width, self.hc, int(self.index["segments"][seg].get("bits", 8)))
            shapes = {"ids": (np.int32, (rows,)), "q": (hidden["q"][0], (rows, hidden["q"][1])),
                      "s": (hidden["s"][0], (rows, hidden["s"][1])), "tp": (np.float16, (rows, self.topk)),
                      "ti": (np.int32, (rows, self.topk))}
            drafts = self.folder / f"seg{seg:05d}.{DR_PART}"
            if self.index.get("draft_rows") and drafts.is_file():
                shapes[DR_PART] = (np.dtype(DR_FIELDS), (rows,))
            self.maps[seg] = {part: np.memmap(self.folder / f"seg{seg:05d}.{part}", dtype=dtype, mode="r",
                                              shape=shape) for part, (dtype, shape) in shapes.items()}
        return self.maps[seg]

    def fetch(self, entry: dict[str, Any]):
        began = time.monotonic()
        held = self.mapped(int(entry["seg"]))
        count = int(entry["count"])
        low = 0 if self.window is None else max(0, int(entry["start"]) - int(self.window))

        def take(part: str) -> torch.Tensor:
            return torch.from_numpy(np.array(held[part][low:count])).to(self.device, non_blocking=True)

        ids = take("ids").long()
        if int(self.index["segments"][int(entry["seg"])].get("bits", 8)) == 4:
            multi = dequantize_rows4(take("q"), take("s"), group_for(self.width, self.hc))
        else:
            multi = dequantize_rows(take("q"), take("s"), self.hc)
        given = (take("tp"), take("ti"))
        if DR_PART in held:
            rows = np.array(held[DR_PART][low:count])
            verdict = np.where(rows["acc"] == DR_NONE, -1.0, (rows["acc"] == 1).astype(np.float32))
            conf = np.where(rows["acc"] == DR_NONE, -1.0, rows["conf"].astype(np.float32))
            given = given + (torch.from_numpy(np.stack([conf, verdict], 1).astype(np.float32)).to(
                self.device, non_blocking=True),)
        self.reads += 1
        self.seconds += time.monotonic() - began
        return ids, multi, int(entry["start"]) - low, given
