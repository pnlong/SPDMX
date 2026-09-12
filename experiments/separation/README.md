# Source separation PoC (ICASSP)

Hybrid Demucs: **Slakh** vs **SPDMX (BDGP-eligible)** vs **both** (union),
matched step budget. Targets: Bass / Drums / Guitar / Piano. Metrics: SI-SDR
on Slakh2100 test + MUSDB bass/drums.

SPDMX is packed only for songs that contain all four targets (same stem makeup
as the Slakh protocol). Prefer a release-tree ``songs.csv`` with
``subset:bdgp`` (from ``synthesis.build_songs_table`` / ``build_spdmx``) so
indexing skips the full corpus; otherwise BDGP is recomputed from ``stems.csv``.
The BDGP-eligible pool is smaller than Slakh train; the ``both`` arm unions
Slakh and SPDMX train/val to test complementarity under the same step budget.

## Setup

```bash
uv pip install demucs
# Slakh path (also in .env as SPDMX_SLAKH_ROOT):
#   /deepfreeze/share/pnlong/slakh2100_flac_redux
# optional MUSDB HQ for cross-domain eval
# export SPDMX_MUSDB_ROOT=/path/to/musdb18hq
```

Default SPDMX root: the **chunked release** `{SPDMX_OUTPUT_DIR}/SPDMX/`
(same layout as Zenodo). Override with `SPDMX_DATASET_ROOT` or `spdmx_root` in
`config.yaml`. Flat `SPDMX_dev/` still works if you point at it.

## Pipeline

```bash
# 1) Remap stems → 4-stem packs (+ mix); BDGP-eligible only
uv run python -m experiments.separation.prepare_stems --corpus both -j 8

# 2) Freeze manifests (slakh | spdmx | both)
uv run python -m experiments.separation.freeze_manifests

# 3) Train — requires GPU; resumes from checkpoints/<arm>/last.ckpt
uv run python -m experiments.separation.train --arm all
# or train only the joint arm: --arm both

# 4) Eval → CSV for paper figures (safe to run arms in parallel)
# Default eval sets: slakh + spdmx_val + musdb; --write-paper →
# `{SPDMX_OUTPUT_DIR}/dev/experiments/separation/eval/separation_sisdr.csv`
uv run python -m experiments.separation.eval --arm all --write-paper
# merge prior per-arm CSVs after parallel runs:
# uv run python -m experiments.separation.eval --merge-only --write-paper
```

Outputs live under `{SPDMX_OUTPUT_DIR}/dev/experiments/separation/` (`packs/`, `manifests/`, `checkpoints/`, `eval/`).
Paper CSV: `{SPDMX_OUTPUT_DIR}/dev/experiments/separation/eval/separation_sisdr.csv`.
Combined figure: `uv run python -m submission.make_figures --only downstream`.
