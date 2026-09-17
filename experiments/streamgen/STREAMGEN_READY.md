# StreamGen is set up

- Clone: `/home/pnlong/spdmx/experiments/streamgen/stream-music-gen`
- SPDMX index: `/home/pnlong/spdmx/analysis/paper_data/streamgen_spdmx_index`
- DAC weights: not downloaded (re-run with `--with-weights`)

## Quick check

```bash
ls /home/pnlong/spdmx/experiments/streamgen/stream-music-gen/stream_music_gen
head -1 /home/pnlong/spdmx/analysis/paper_data/streamgen_spdmx_index/spdmx_multitrack.jsonl
```

## Next (upstream)

1. Follow `/home/pnlong/spdmx/analysis/paper_data/streamgen_spdmx_index/ADAPTER.md` to add an SPDMX dataset class.
2. Dump DAC / RMS / mixdown features (needs the DAC weights).
3. Train the pilot:

```bash
cd /home/pnlong/spdmx/experiments/streamgen/stream-music-gen
python scripts/train_dec_online.py \
  --args.load configs/online/decoder_online_future_visibility_0.yml \
  --save_dir logs/dec_online_future_visibility_0
```

Record metrics into `analysis/paper_data/streamgen_metrics.csv` when done.

## Notes

- Setup skips upstream `torch`/`torchaudio` pins so your SPDMX torch stays.
- If you also installed YourMT3, `transformers` versions may conflict; use a separate venv if training both in parallel.
