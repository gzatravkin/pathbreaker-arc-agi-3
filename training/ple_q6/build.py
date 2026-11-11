import base64
import hashlib
import json
import sys
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
COUNT = 3
MODEL = "dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/Transformers/default/1"
DF11 = "grigoriyzatravkin/arc3-w4a16-ple-df11"
IMAGE = "gcr.io/kaggle-images/python@sha256:dafd4ce5668bbf1ad422e4c109e0f18c9623c3a7c7f48b0235f13142755c40b9"


def main():
    raw = (ROOT / "serving" / "dataset" / "ple_hot.py").read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    packed = base64.b64encode(zlib.compress(raw, 9)).decode("ascii")
    template = (HERE / "template.py").read_text(encoding="utf-8")
    for index in range(COUNT):
        out = HERE / f"part{index}"
        out.mkdir(exist_ok=True)
        slug = f"arc3-ple-q6-bake-{index}"
        script = (template.replace("@@INDEX@@", str(index)).replace("@@COUNT@@", str(COUNT))
                  .replace("@@SOURCE_SHA@@", sha).replace("@@SOURCE@@", packed))
        (out / f"{slug}.py").write_text(script, encoding="utf-8", newline="\n")
        meta = {"id": f"grigoriyzatravkin/{slug}", "title": f"arc3 ple q6 bake {index}", "code_file": f"{slug}.py",
                "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": False,
                "enable_tpu": False, "enable_internet": False, "keywords": [], "dataset_sources": [DF11],
                "kernel_sources": [], "competition_sources": [], "model_sources": [MODEL], "docker_image": IMAGE}
        (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=1), encoding="utf-8", newline="\n")
        print(out, sha[:12], len(script))


if __name__ == "__main__":
    sys.exit(main())
