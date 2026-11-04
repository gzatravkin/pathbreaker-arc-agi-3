# pathbreaker — ARC Prize 2026, ARC-AGI-3

The code of our two selected final submissions to the Kaggle competition
[ARC Prize 2026 — ARC-AGI-3](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3).
Public leaderboard at the close: rank 6, score 54.55. The private leaderboard decides the prizes.

**Write-up:** [WRITEUP.md](WRITEUP.md) — how a 512-expert agent runs at 110 concurrent sessions on one GPU, which
experts and which context tokens the model itself uses, and the live switch from speed to quality inside a run.
The Paper Track version is [docs/paper-track.md](docs/paper-track.md).

| | submission | public score | where |
|---|---|---|---|
| 1 | kernel `franzen-bulletk400hb1101`, version 2, submitted 2026-11-01 (Kaggle ref 57886577): the agent and serve of submission 2 plus a live expert switch from 400 to 512 experts at play minute 360, 45 sessions after the switch, and a scheduler that parks the copies with the largest share of their game done | **54.55** | repository root: `notebook/`, `bundle/`, `serving/` |
| 2 | kernel `franzen-bulletk400vh1104`, version 1, submitted 2026-10-31 (Kaggle ref 57853423): Franzen's milestone-2 agent with our changes on the clean serve, 400 experts, 110 sessions, 540 minutes | 50.48 | `submissions/vh1104-v1/` |

Each is a snapshot of exactly what ran, taken from the Kaggle notebook and the dataset versions it mounted. No
development history is included. An earlier snapshot of this repository (commit `52fe509`) held a later,
non-selected submission (`franzen-bulletk400fin1` v5, 41.34).

## Layout (the same for both submissions)

| Path | What it is |
|---|---|
| `notebook/` | The submission notebook and its `kernel-metadata.json` (inputs, accelerator) |
| `bundle/tufa_bundle/src/ARC3-Inference/` | The agent harness: The Duck (Tufa Labs), with our changes |
| `bundle/tufa_bundle/src/tufa-arc-agi-framework/` | TAAF, the game/benchmark framework (Tufa Labs), with our changes |
| `bundle/harness_presets/` | The `current` and `legacy` harness presets the notebook selects from |
| `bundle/serve/` | Serve setup scripts; `pureallk400fast_setup.sh` holds every serve knob |
| `bundle/stress_games/` | Synthetic games used to check the harness offline |
| `bundle/drift_turns.jsonl.gz.bin` | Recorded agent requests on a public game, a data file shipped in the bundle |
| `bundle/MANIFEST.json` | The bundle's own manifest |
| `serving/` | The serving layer: our vLLM patches and kernels, `serving_setup.py`, the compiled `w4a8fix_op.so`; `expert_hot.py` is the live expert switch (submission 1 only) |
| `docs/figures/` | Figures of the write-up |

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
