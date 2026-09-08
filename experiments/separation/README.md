# Source separation PoC (ICASSP)

Hybrid Demucs quantity ablation: **Slakh** vs **sPDMX-matched** vs **sPDMX-full**.
Targets: Bass / Drums / Guitar / Piano. Metrics: SI-SDR on Slakh2100 test + MUSDB bass/drums.

## Setup

```bash
uv pip install demucs
# Slakh path (also in .env as SPDMX_SLAKH_ROOT):
#   /deepfreeze/share/pnlong/slakh2100_flac_redux
# optional MUSDB HQ for cross-domain eval
# export SPDMX_MUSDB_ROOT=/path/to/musdb18hq
```

Default sPDMX root: the **chunked release** `{SPDMX_OUTPUT_DIR}/SPDMX/`
(same layout as Zenodo). Override with `SPDMX_DATASET_ROOT` or `spdmx_root` in
`config.yaml`. Flat `SPDMX_dev/` still works if you point at it.

## Pipeline

One-shot (GPU host for train/eval):

```bash
./experiments/separation/run_poc.sh
# SAO: SAO_PRETRAINED_CKPT=/path/to/sao.ckpt ./experiments/separation/run_poc.sh
```

Or step-by-step:

```bash
# 1) Remap stems → 4-stem packs (+ mix); -j parallelizes decode/encode
uv run python -m experiments.separation.prepare_stems --corpus both -j 8

# 2) Freeze manifests (matched hours ≈ Slakh train)
uv run python -m experiments.separation.freeze_manifests

# 3) Train (matched step budget from config.yaml) — requires GPU
#    Auto-resumes from checkpoints/<arm>/last.ckpt (model+optimizer+step).
#    Use --reset to start fresh. Logs append to losses.csv / losses.jsonl.
uv run python -m experiments.separation.train --arm all

# 4) Eval → CSV for paper figures
uv run python -m experiments.separation.eval --arm all --write-paper
```

Outputs live under `{SPDMX_OUTPUT_DIR}/dev/experiments/separation/` (`packs/`, `manifests/`, `checkpoints/`, `eval/`).
Paper CSV: `submission/data/separation_sisdr.csv`. Combined figure: `uv run python -m submission.make_figures --only downstream`.
