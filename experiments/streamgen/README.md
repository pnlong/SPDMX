# StreamGen — streaming accompaniment scale-up

**Question:** Does training [stream-music-gen](https://github.com/lukewys/stream-music-gen)
(Wu et al., 2025) on SPDMX multitrack beat the Slakh baseline under matched steps?

## One-command setup

From the repo root:

```bash
uv run python -m experiments.streamgen.setup_streamgen
```

That will:

1. Clone `lukewys/stream-music-gen` into `experiments/streamgen/stream-music-gen`
2. Install filtered Python deps into your current env (keeps your existing torch)
3. Build / reuse the SPDMX JSONL index under `analysis/paper_data/streamgen_spdmx_index/`
4. Write `experiments/streamgen/STREAMGEN_READY.md` with next steps

Optional flags:

```bash
# Also download the causal DAC checkpoint (needed before dataset dump)
uv run python -m experiments.streamgen.setup_streamgen --with-weights

# Also install evaluation extras (COCOLA / FAD / …)
uv run python -m experiments.streamgen.setup_streamgen --with-eval

# Re-clone from scratch
uv run python -m experiments.streamgen.setup_streamgen --force

# Code already cloned; only refresh deps + index
uv run python -m experiments.streamgen.setup_streamgen --skip-clone
```

## Pilot

- Config: `dec_online`, `future_visibility = 0`
- Arms: Slakh | SPDMX multitrack | optional union / hour-matched SPDMX
- Eval: Slakh-test COCOLA / Beat F1 / FAD; secondary SPDMX held-out

## After setup

```bash
# Index only (if you need to rebuild)
uv run python -m experiments.streamgen.prepare_spdmx_index

# Smoke (first N songs)
uv run python -m experiments.streamgen.prepare_spdmx_index --max-songs 100

# Placeholder metrics for paper/blog
uv run python -m experiments.streamgen.train --write-placeholder-metrics
```

Training still uses upstream `scripts/train_dec_online.py` — see `STREAMGEN_READY.md`
and `{index}/ADAPTER.md` after setup. Blog: `docs/blog/streamgen.html`.
