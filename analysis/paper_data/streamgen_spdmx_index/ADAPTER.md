# SPDMX adapter for stream-music-gen

Source of truth: `experiments/streamgen/adapters/spdmx.py`
(installed into the upstream clone as `stream_music_gen/dataset/spdmx.py`).

## One-command data prep

```bash
# Slakh baseline (default)
bash experiments/streamgen/prepare_streamgen_data.sh --gpu 1

# SPDMX arm (rebuilds JSONL index, wires adapter)
bash experiments/streamgen/prepare_streamgen_data.sh --datasets spdmx --gpu 1 --rebuild-index

# Both
bash experiments/streamgen/prepare_streamgen_data.sh --datasets slakh2100,spdmx --gpu 1
```

That script runs: index (SPDMX) → DAC extract → RMS → mixdown dump.

## Manual adapter wiring

1. Build index: `uv run python -m experiments.streamgen.prepare_spdmx_index`
2. Layout under `stream_music_gen_data/spdmx/`:
   - `spdmx_multitrack.jsonl` → paper_data index
   - `audio` → `$SPDMX_DATASET_ROOT` (default `/deepfreeze/share/SPDMX/SPDMX`)
3. Copy adapter + register `spdmx` in `constants.DATASET_SPLITS` and extract/dump scripts
   (handled by `prepare_streamgen_data.sh`).
4. Extract / dump with `--datasets spdmx`.
5. Train `dec_online` with `dataset_names: [spdmx]` (or matched-step Slakh vs SPDMX).
6. Eval on Slakh test via `scripts/gen_pred/gen_and_evaluate.py`.
