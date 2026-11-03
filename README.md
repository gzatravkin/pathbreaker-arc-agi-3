# pathbreaker — ARC Prize 2026, ARC-AGI-3

The code of our final submission to the Kaggle competition
[ARC Prize 2026 — ARC-AGI-3](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3):
the kernel `franzen-bulletk400fin1`, version 5, submitted 2026-11-02.
Public leaderboard rank at the close: 6 (the private leaderboard decides the prizes).

This is a single snapshot of exactly what ran, taken from the Kaggle notebook and the
dataset versions it mounted. No development history is included.

## Layout

| Path | What it is |
|---|---|
| `notebook/` | The submission notebook and its `kernel-metadata.json` (inputs, accelerator) |
| `bundle/tufa_bundle/src/ARC3-Inference/` | The agent harness: The Duck (Tufa Labs), with our changes |
| `bundle/tufa_bundle/src/tufa-arc-agi-framework/` | TAAF, the game/benchmark framework (Tufa Labs), with our changes |
| `bundle/harness_presets/` | The `current` and `legacy` harness presets the notebook selects from |
| `bundle/serve/` | Serve setup scripts |
| `bundle/stress_games/` | Synthetic games used to check the harness offline |
| `bundle/drift_turns.jsonl.gz.bin` | Recorded agent requests on a public game, a data file shipped in the bundle |
| `bundle/MANIFEST.json` | The bundle's own manifest |
| `serving/` | The serving layer: our vLLM patches and kernels, `serving_setup.py`, the compiled `w4a8fix_op.so` |

## How it ran

- Accelerator: `NvidiaRtxPro6000` (see `notebook/kernel-metadata.json`), no internet, local model only.
- Model: Qwen3.8 Flash Next, W4A16 AutoRound, from the Kaggle model
  `dfranzen/intel-qwen3.8-flash-next-w4a16-autoround`; draft model
  `dfranzen/albucino-qwen3-8-flash-next-drafter`.
- Serve: vLLM 0.30 runtime (Kaggle dataset `arc3-vllm030-runtime-flat`, not included: 6.9 GB of
  third-party binaries) with the patches in `serving/`.
- The other inputs listed in `kernel-metadata.json` are weights (MTP heads, drafter, quantized rows)
  and start caches (compile and pyc seeds). They are not included here.

To reproduce, attach the inputs listed in `notebook/kernel-metadata.json` to the notebook and run it
in the competition's Kaggle environment.

## License

CC BY 4.0 (see `LICENSE`), as the competition rules require of winning submissions.
Third-party parts keep their authors; see `NOTICE.md`.
