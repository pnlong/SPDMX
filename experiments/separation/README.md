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
## Multi-stem stem vs other (full SPDMX)

Separate from the BDGP / Slakh comparison: 2-source HTDemucs on songs with
≥2 stems (~88k). Target = one stem; other = mix − stem.

**v2 training recipe** (`config_multistem.yaml`): stem-weighted L1
(`stem:other = 5:1`), differentiable −SI-SDR on the stem source
(`sisdr_loss_weight: 0.5`), energy-aware stem/crop sampling
(`stem_min_energy_ratio: 0.05`), and `max_steps: 96000`. Plain L1 at 32k
steps collapsed to the mixture; keep a backup of that run as
`checkpoints/multistem_v1_baseline/` if you reset.

```bash
# 1) Freeze train/val manifests from songs.csv
uv run python -m experiments.separation.freeze_multistem -j 32

# 2) Train (GPU). Use --reset after backing up v1 checkpoints.
uv run python -m experiments.separation.train \
  --arm multistem \
  --config experiments/separation/config_multistem.yaml \
  --reset

# 3) Eval SI-SDR on held-out multi-stem split
uv run python -m experiments.separation.eval_multistem --write-paper
```

Checkpoints: `{SPDMX_OUTPUT_DIR}/dev/experiments/separation/checkpoints/multistem/`.
Success check: stem mean SI-SDR > 0 dB and clearly above `mixture_si_sdr`
(aim ≥ +2 dB improvement). If loss is still flat by ~20–30k steps, raise
stem / SI-SDR weights rather than burning the full budget blindly.