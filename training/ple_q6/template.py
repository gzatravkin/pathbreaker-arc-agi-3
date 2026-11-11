import base64
import importlib.util
import json
import os
import shutil
import sys
import time
import zlib
from pathlib import Path

INDEX = @@INDEX@@
COUNT = @@COUNT@@
SOURCE_SHA = "@@SOURCE_SHA@@"
SOURCE = "@@SOURCE@@"


def say(text):
    print(f"{time.strftime('%H:%M:%S')} {text}", flush=True)


def main():
    import torch

    torch.set_num_threads(max(1, os.cpu_count() or 1))
    helper = Path("/tmp/taaf_ple_hot.py")
    helper.write_bytes(zlib.decompress(base64.b64decode(SOURCE)))
    spec = importlib.util.spec_from_file_location("taaf_ple_hot", helper)
    hot = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = hot
    spec.loader.exec_module(hot)
    say(f"ple_hot source {SOURCE_SHA[:12]} loaded; torch {torch.__version__}, {torch.get_num_threads()} threads")
    indexes = [path for path in Path("/kaggle/input").glob("**/model.safetensors.index.json")
               if "ngram_embedding" in path.read_text(encoding="utf-8")]
    if len(indexes) != 1:
        raise SystemExit(f"expected one checkpoint with PLE shards under /kaggle/input, found {indexes}")
    directory = indexes[0].parent
    df11 = hot.find_df11_file()
    if df11 is None:
        raise SystemExit("the DF11 dataset is not mounted, so its recorded probe rows cannot be checked")
    probes = hot.read_df11_header(df11)["probes"]
    free = shutil.disk_usage("/kaggle/working").free
    say(f"checkpoint {directory}; DF11 {df11} with {len(probes)} probe rows; /kaggle/working free "
        f"{free / 2**30:.1f} GiB")
    path, header = hot.q6_bake(directory, INDEX, COUNT, "/kaggle/working", kind="int6", df11_probes=probes, say=say)
    again = hot.q6_read_header(path)
    if again["probes"] != header["probes"] or path.stat().st_size != hot.q6_layout(again)[2]:
        raise SystemExit("the part does not read back its own header")
    summary = {key: header[key] for key in ("kind", "rows", "width", "first", "end", "index", "count", "fingerprint",
                                            "df11_checked", "df11_probes", "size")}
    summary["probes"] = len(header["probes"])
    summary["source_sha256"] = SOURCE_SHA
    Path("/kaggle/working/ple_q6_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    say(f"done: {json.dumps(summary)}")


main()
