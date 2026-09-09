# Lite Stable Audio Open fine-tune PoC

sPDMX is a **large MIDI-paired audio corpus**. Separation uses a BDGP-eligible
subset; SAO compares that subset to the full corpus.

| Arm | Data |
|-----|------|
| **slakh** | Slakh train mixes |
| **spdmx_matched** | Songs with ``subset:bdgp`` in `songs.csv` (same BDGP pool as separation) |
| **spdmx_full** | All sPDMX songs (`subset:all`) |

Eval: OpenL3/VGGish **FAD** + **CLAP** (no listening).
`prepare_dataset` indexes mixes directly (Slakh `mix.flac`; sPDMX stem-sums
cached under `{SPDMX_OUTPUT_DIR}/dev/experiments/sao/datasets/mixes/`).
Prefer building `songs.csv` first so matched uses ``subset:bdgp``:

```bash
uv run python -m synthesis.build_songs_table
uv run python -m experiments.sao.prepare_dataset -j 8
```

```bash
git clone https://github.com/Stability-AI/stable-audio-tools experiments/sao/stable-audio-tools
uv pip install -e "experiments/sao/stable-audio-tools[train]"
uv pip install laion-clap frechet_audio_distance  # metrics
```

## Pipeline

```bash
# Index Slakh + sPDMX; matched = BDGP pool; full = all (-j workers)
uv run python -m experiments.sao.prepare_dataset -j 8

uv run python -m experiments.sao.train --arm all --pretrained-ckpt /path/to/sao.ckpt
uv run python -m experiments.sao.generate --arm all
uv run python -m experiments.sao.metrics --ref-dir /path/to/heldout_mixes
```

Outputs: `{SPDMX_OUTPUT_DIR}/dev/experiments/sao/`. Paper CSV: `submission/data/sao_metrics.csv`.

See also [`CHECKPOINT.md`](CHECKPOINT.md) for downloading SAO 1.0 weights.
