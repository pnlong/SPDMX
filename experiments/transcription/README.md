# Transcription C1 — multi-instrument AMT scale-up

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

## After setup

```bash
# Manifests only (if you need to rebuild)
uv run python -m experiments.transcription.prepare_manifest

# Placeholder metrics CSV for the paper/blog
uv run python -m experiments.transcription.train --write-placeholder-metrics
```

Training still uses YourMT3’s own CLI inside `YourMT3/amt/src` — see `YOURMT3_READY.md`
after setup. Blog: `docs/blog/transcription.html`.
