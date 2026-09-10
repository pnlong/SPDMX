# Lite Stable Audio Open fine-tune PoC

sPDMX is a **large MIDI-paired audio corpus**. Separation uses a BDGP-eligible
subset; SAO compares that subset to the full corpus.

| Arm | Data |
|-----|------|
| **slakh** | Slakh train mixes |
| **spdmx_matched** | Songs with ``subset:bdgp`` in `songs.csv` (same BDGP pool as separation) |
| **spdmx_full** | All sPDMX songs with a shipped mix (`subset:all`) |

Eval: OpenL3/VGGish **FAD** + **CLAP** (no listening).
`prepare_dataset` indexes **shipped** sPDMX mixes (`chunk_N/<song_id>/mix.flac`
or `SPDMX_dev/mix/<song_id>.flac`) the same way Slakh uses native `mix.flac` —
no stem re-sum. Prefer building `songs.csv` first so matched uses ``subset:bdgp``.

**Channels:** sPDMX mixes are **mono** (same as stems). SAO’s model is stereo
(`audio_channels: 2`); `stable-audio-tools` applies its ``Stereo()`` transform at
load time (duplicate L/R). Do not convert mixes to stereo in the dataset.

```bash
# Ensure songs.csv has song_length (mix FLAC duration). After packaging:
#   uv run python -m synthesis.build_songs_table --enrich-lengths-only -j 32
uv run python -m experiments.sao.prepare_dataset
```

```bash
git clone https://github.com/Stability-AI/stable-audio-tools experiments/sao/stable-audio-tools
uv pip install -e "experiments/sao/stable-audio-tools[train]"
uv pip install laion-clap frechet_audio_distance  # metrics
```

## Pipeline

```bash
# Index Slakh + sPDMX (reads songs.csv song_length / subset:bdgp when present)
uv run python -m experiments.sao.prepare_dataset

uv run python -m experiments.sao.train --arm all --pretrained-ckpt /path/to/sao.ckpt
uv run python -m experiments.sao.generate --arm all
uv run python -m experiments.sao.metrics --ref-dir /path/to/heldout_mixes
```

Outputs: `{SPDMX_OUTPUT_DIR}/dev/experiments/sao/`. Paper CSV: `submission/data/sao_metrics.csv`.

See also [`CHECKPOINT.md`](CHECKPOINT.md) for downloading SAO 1.0 weights.
