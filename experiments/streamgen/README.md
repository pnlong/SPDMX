# StreamGen — streaming accompaniment scale-up

**Question:** Does training [stream-music-gen](https://github.com/lukewys/stream-music-gen)
(Wu et al., 2025) on SPDMX multitrack beat the Slakh baseline under matched steps?

## One-command setup

```bash
uv run python -m experiments.streamgen.setup_streamgen
# optional: --with-weights  (or use existing DAC under jazz-standard-dataset)
```

## One-command data prep (DAC → RMS → mixdown)

```bash
# Slakh baseline on GPU 1 (reuses deepfreeze 44.1 kHz FLAC)
bash experiments/streamgen/prepare_streamgen_data.sh --gpu 1

# SPDMX arm
bash experiments/streamgen/prepare_streamgen_data.sh --datasets spdmx --gpu 1 --rebuild-index

# Smoke
bash experiments/streamgen/prepare_streamgen_data.sh --datasets spdmx --max-songs 50 --skip-dump --gpu 1

# DAC chunking (VRAM / util): --win-duration SECS --chunk-batch-size N
bash experiments/streamgen/prepare_streamgen_data.sh --gpu 1 --chunk-batch-size 8
```

SPDMX adapter: `experiments/streamgen/adapters/spdmx.py` (see index `ADAPTER.md`).

## Pilot

- Config: `dec_online`, `future_visibility = 0`
- Arms: Slakh | SPDMX multitrack | optional union / hour-matched SPDMX
- Eval: Slakh-test COCOLA / Beat F1 / FAD; secondary SPDMX held-out

## After data prep — train

```bash
cd experiments/streamgen/stream-music-gen
CUDA_VISIBLE_DEVICES=1 uv run --project ../.. \
  python scripts/train_dec_online.py \
  --args.load configs/online/decoder_online_future_visibility_0.yml \
  --save_dir /deepfreeze/share/SPDMX/dev/experiments/streamgen/logs/dec_online_fv0_slakh \
  --batch_size 8 --precision 16-mixed
```

Placeholder metrics: `uv run python -m experiments.streamgen.train --write-placeholder-metrics`

Blog: `docs/blog/streamgen.html`.
