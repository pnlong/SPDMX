# Transcription — multi-instrument AMT scale-up

**Question:** Holding a multi-instrument AMT model fixed, does more multitrack
data (SPDMX) beat Slakh2100-redux on Slakh-test note F1?

Framing follows [MT3](https://github.com/magenta/mt3) / [YourMT3](https://github.com/mimbres/YourMT3)
(runnable code is the [HF Spaces pre-release](https://huggingface.co/spaces/mimbres/YourMT3)).

## One-command setup

From the repo root:

```bash
uv run python -m experiments.transcription.setup_yourmt3
```

That will:

1. `git lfs install` + clone YourMT3 into `experiments/transcription/YourMT3`
2. Install filtered Python deps into your current env (keeps your existing torch)
3. Build / reuse manifests under `analysis/paper_data/transcription_manifests/`
4. Write `experiments/transcription/YOURMT3_READY.md` with next steps

Optional flags:

```bash
# Also download checkpoint blobs (large)
uv run python -m experiments.transcription.setup_yourmt3 --with-weights

# Re-clone from scratch
uv run python -m experiments.transcription.setup_yourmt3 --force

# Code already cloned; only refresh deps + manifests
uv run python -m experiments.transcription.setup_yourmt3 --skip-clone
```

## Arms

| Arm | Data |
|-----|------|
| `slakh` | Slakh2100-redux (no `omitted/`) |
| `spdmx` | SPDMX songs with `n_tracks >= 2` |
| `both` | Union |
| `spdmx_hour_matched` | Optional ~145 h SPDMX subsample |

## After setup — build YourMT3 indexes + train

Convert CSV manifests → `yourmt3_indexes` (16 kHz WAV + note caches):

```bash
# Full (Slakh + SPDMX). Writes under deepfreeze …/transcription/yourmt3_data
uv run python -m experiments.transcription.manifest_to_yourmt3_indexes

# Smoke
uv run python -m experiments.transcription.manifest_to_yourmt3_indexes --arms spdmx --max-songs 20
```

Then train:

```bash
# Slakh arm (GPU 1)
uv run python -m experiments.transcription.train --arm slakh --gpu 1

# SPDMX arm (GPU 2)
uv run python -m experiments.transcription.train --arm spdmx --gpu 2
```

This wraps YourMT3's `amt/src/train.py`. The first positional arg there is only an
experiment id (checkpoints / W&B); the wrapper defaults it to the arm name.
Pass extras after `--`, e.g. `-- --precision 32`.

Presets added by the converter: `slakh_redux`, `spdmx`, `spdmx_hm`, `slakh_spdmx`, multi `slakh_vs_spdmx`.
