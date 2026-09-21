# Source separation PoC (ICASSP)

Hybrid Demucs: **Slakh** vs **SPDMX (BDGP-eligible)** vs **both** (union),
matched step budget. Targets: Bass / Drums / Guitar / Piano. Metrics: SI-SDR
on Slakh2100 test + MUSDB bass/drums + MedleyDB (V1+V2) + MoisesDB (when present).

SPDMX is packed only for songs that contain all four targets (same stem makeup
as the Slakh protocol). Prefer a release-tree ``songs.csv`` with
``subset:bdgp`` (from ``synthesis.build_songs_table`` / ``build_spdmx``) so
indexing skips the full corpus; otherwise BDGP is recomputed from ``stems.csv``.
The BDGP-eligible pool is smaller than Slakh train; the ``both`` arm unions
Slakh and SPDMX train/val to test complementarity under the same step budget.

## Setup

```bash
uv pip install demucs
# optional MoisesDB Python API (eval soft-skips if missing):
# uv pip install "git+https://github.com/moises-ai/moises-db.git"
# Slakh path (also in .env as SPDMX_SLAKH_ROOT):
#   /deepfreeze/share/pnlong/slakh2100_flac_redux
# optional real-data eval roots:
# export SPDMX_MUSDB_ROOT=/path/to/musdb18hq
# export SPDMX_MEDLEYDB_ROOT=/deepfreeze/share/pnlong/MedleyDB   # V1/ + V2/
# export SPDMX_MOISESDB_ROOT=/deepfreeze/share/pnlong/MoisesDB   # moisesdb_v0.1/<uuid>/
```

### Real-data layouts

- **MUSDB18-HQ:** `test/<track>/{bass,drums,mixture}.wav` (bass+drums only).
- **MedleyDB:** `{SPDMX_MEDLEYDB_ROOT}/{V1,V2}/<Artist_Track>/` with `*_MIX.wav` and
  `*_STEMS/*_STEM_NN.wav`. Instrument labels come from YAML under
  `{SPDMX_MEDLEYDB_ROOT}/Metadata/` (or `SPDMX_MEDLEYDB_METADATA`), typically
  `/deepfreeze/share/pnlong/MedleyDB/Metadata`.
  Tracks with `has_bleed: yes` are skipped. Present BDGP targets only are scored.
- **MoisesDB:** unpack so `{SPDMX_MOISESDB_ROOT}/moisesdb_v0.1/<uuid>/` exists.
  Eval uses `moisesdb.dataset.MoisesDB` and scores present `bass`/`drums`/`guitar`/`piano`
  stems. Soft-skips if the root is missing/empty or the package is not installed.

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
# Default eval sets: slakh + spdmx_val + musdb + medleydb + moisesdb;
# --write-paper → `{SPDMX_OUTPUT_DIR}/dev/experiments/separation/eval/separation_sisdr.csv`
uv run python -m experiments.separation.eval --arm all --write-paper
# real-data only (re-run just new sets):
# uv run python -m experiments.separation.eval --arm all \
#   --eval-sets musdb medleydb moisesdb --write-paper
# merge prior per-arm CSVs after parallel runs:
# uv run python -m experiments.separation.eval --merge-only --write-paper
```

Outputs live under `{SPDMX_OUTPUT_DIR}/dev/experiments/separation/` (`packs/`, `manifests/`, `checkpoints/`, `eval/`).
Paper CSV: `{SPDMX_OUTPUT_DIR}/dev/experiments/separation/eval/separation_sisdr.csv`.
## Multi-stem stem vs other (full SPDMX)

Separate from the BDGP / Slakh comparison: 2-source HTDemucs on songs with
≥2 stems (~88k). Target = one stem; other = mix − stem.

**v3 training recipe** (`config_multistem.yaml`): stem-weighted L1
(`stem:other = 8:1`), differentiable −SI-SDR on the stem source
(`sisdr_loss_weight: 1.0`), energy-aware stem/crop sampling on **train and
val** (`stem_min_energy_ratio: 0.05`; val searches tracks × crop grid),
mix rebuilt as the stem sum (`rebuild_mix_from_stems: true`), and
`max_steps: 96000`. Plain L1 at 32k steps collapsed to the mixture; v2 used
weaker weights and val=`tracks[0]`@t=0 (silent intros), so keep backups under
`checkpoints/multistem_v1_baseline/` / `…_v2_*` if you reset.

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
(aim ≥ +2 dB improvement). If loss is still flat by ~20–30k steps after a
v3 `--reset`, raise stem / SI-SDR weights further rather than burning the
full budget blindly.
