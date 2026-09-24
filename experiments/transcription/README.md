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

Defaults are **step-based**: `--max-steps 100000`, validate every `--val-interval 2000` steps
(not once per epoch), and `--limit-val-batches 32` so each val pass stays short.
Override via flags or `experiments/transcription/config.yaml`.

**Resume:** re-run the same command. Checkpoints live under
`YourMT3/amt/logs/transcription/<exp-id>/checkpoints/last.ckpt`
(`--exp-id` defaults to the arm name). A matching `last.ckpt` is loaded automatically
(optimizer + step). `last.ckpt` is rewritten every `--val-interval` steps.

This wraps YourMT3's `amt/src/train.py`. The first positional arg there is only an
experiment id (checkpoints / W&B); the wrapper defaults it to the arm name.
Pass extras after `--`, e.g. `-- --precision 32`.

Presets added by the converter: `slakh_redux`, `spdmx`, `spdmx_hm`, `slakh_spdmx`, multi `slakh_vs_spdmx`.

## Eval with per-track bootstrap CIs

Checkpoints are expected under
`{SPDMX_OUTPUT_DIR}/dev/experiments/transcription/checkpoints/{slakh,spdmx}/`.
Eval dumps one row per song, then percentile-bootstrap 95% CIs (mean ± half-width
for the paper CSV).

**Overnight (recommended, 4 GPUs):** SPDMX→SPDMX on GPUs 0–1; the two Slakh-test
jobs on GPUs 2 and 3 in parallel.

```bash
# from repo root; leave running overnight
uv run python -m experiments.transcription.eval \
  --all --gpu 0,1,2,3 --parallel --write-paper \
  --subbsz 128 --num-workers 8
```

Logs / JSONL / `summary.json` land under
`{SPDMX_OUTPUT_DIR}/dev/experiments/transcription/eval/<train>__<preset>/`
(e.g. `spdmx__spdmx/eval.log`). Paper CSV:
`analysis/paper_data/transcription_note_f1.csv`.

**Single job** (e.g. only the large SPDMX test set):

```bash
uv run python -m experiments.transcription.eval \
  --train-arm spdmx --test-preset spdmx --gpu 0,1 --write-paper
```

**Recompute CIs / paper CSV after a run** (CPU only):

```bash
uv run python -m experiments.transcription.eval --merge-only --write-paper
```

Throughput knobs (also set automatically by the wrapper):

| Env / flag | Default | Role |
|---|---|---|
| `--subbsz` / `SPDMX_TEST_SUBBSZ` | 256 | Chunk size inside each packed `inference_file` call |
| `--pack-target-segs` / `SPDMX_PACK_TARGET_SEGS` | 256 | Soft target; used with max for denser best-fit packing |
| `SPDMX_PACK_MAX_SEGS` | `max(512, 2×target)` | Hard bin capacity (best-fit decreasing); raises Slakh fill from ~1 song/step to ~512 segs/step |
| `SPDMX_EVAL_SKIP_TOKENS` | `1` | Skip GT tokenization at test (unused by `inference_file`; otherwise CPU-starves the GPU) |
| `--num-workers` / `SPDMX_TEST_NUM_WORKERS` | 8 | DataLoader workers **per GPU** |
| `--gpu` + DDP | — | Multi-GPU file sharding |
| `--precision` | `bf16-mixed` | Amp |

**Why util stayed low with only `SPDMX_TEST_SUBBSZ`:** eval is still one song per step; a typical SPDMX song has ~30 segments, so raising subbsz past that does nothing. Packing (~4 songs → ~225 segments/step) is required.

**Other machine:** `YourMT3/` is gitignored. After pull, install overlays (packing + skip-tokenize + pack consumer) with:

```bash
uv run python -c "from experiments.transcription.patch_yourmt3 import apply_yourmt3_patches; print(apply_yourmt3_patches())"
```

Startup must print `[SPDMX] PackedAudioFileDataset ...` and `[SPDMX] test_step batch=0 n_files=... n_segs=... packed=True`. If `packed=False` / `n_segs≈30`, packing is not live.
