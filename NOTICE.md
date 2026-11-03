# Notice

This repository contains modified versions of third-party work.

- **The Duck** (ARC3-Inference harness) and **TAAF** (tufa-arc-agi-framework) — Tufa Labs,
  https://github.com/Tufalabs/duck-harness — the winning solution to ARC-AGI-3 Milestone 1;
  the competition rules license winning submissions under CC BY 4.0. Modified here.
- **Models** used at run time (not included): Kaggle models
  `dfranzen/intel-qwen3.8-flash-next-w4a16-autoround` and
  `dfranzen/albucino-qwen3-8-flash-next-drafter`.
- **vLLM** — https://github.com/vllm-project/vllm (Apache-2.0). The runtime is not included;
  `serving/` holds patches against it.
- `serving/wheels/zstandard-*.whl` — python-zstandard (BSD-3-Clause), redistributed unmodified.
