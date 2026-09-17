# ICASSP pilots ready

- SPDMX release: `/deepfreeze/share/SPDMX/SPDMX`
- Slakh: `/deepfreeze/share/pnlong/slakh2100_flac_redux`
- Output root: `/deepfreeze/share/SPDMX`

## Per-experiment next steps

- Transcription (YourMT3): `experiments/transcription/YOURMT3_READY.md`
- StreamGen: `experiments/streamgen/STREAMGEN_READY.md`

## Train stubs (metrics placeholders)

```bash
uv run python -m experiments.transcription.train --write-placeholder-metrics
uv run python -m experiments.streamgen.train --write-placeholder-metrics
```

## Note on deps

YourMT3 and stream-music-gen pin different `transformers` ranges.
This setup installs **StreamGen last**. If a YourMT3 train breaks on
`transformers`, re-run:

```bash
uv run python -m experiments.transcription.setup_yourmt3 --skip-clone
```

Or use separate venvs for the two pilots.
