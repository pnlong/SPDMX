# synthesis

Turn PDMX symbolic MIDI into mono FLAC stems; optionally realify with Stable Audio 3.

Environment setup: **[`SETUP.md`](../SETUP.md)** at repo root.

Production multi-machine setup: **[`FINAL_SETUP.md`](FINAL_SETUP.md)**.

See also [`RENDERING_NOTES.md`](RENDERING_NOTES.md) for ablation design, Slakh alignment, and output layout.
Paper-oriented mixing / stem-summability write-up: [`MIXING.md`](MIXING.md).

## Before you synthesize

Run **GM register correction** first (track-name ↔ GM id fixes). Synthesize loads
`{OUTPUT_DIR}/dev/analysis/instruments/all_valid/register.csv` by default and errors if it is missing.

```bash
uv run python -m analysis.prepare_synthesis --subset all_valid -j 8
```

Flags: `--register PATH` to point at another CSV; `--no-register` to use raw MIDI programs.

## Entry points

| Script | Purpose |
|--------|---------|
| `synthesize.py` | Main CLI: ablation sample (default) or `--full` PDMX, `--render-mode`, `--realify` |
| `mix.py` | Post-hoc LUFS × velocity dynamics × peak gain so stems remain summable |
| `listening/serve.py` | Localhost viewer for A1–CB2 ablation comparison |
| `listening/make_clips.py` | Aligned 10s clips (windows from A1) for listening |
| `build_spdmx.py` | Post-render: build `{OUTPUT}/SPDMX/` chunks from flat `SPDMX_dev/` (no mutate) |
| `build_songs_table.py` | Song-level `songs.csv` + `subset:*` from `stems.csv` (also invoked by `build_spdmx`) |
| `distribute_spdmx.py` | Stage Zenodo files (metadata + `chunk_NNN.zip`) |

## Source files

| File | Description |
|------|-------------|
| `synthesize.py` | MIDI → fluidsynth → mono FLAC stems; optional SA3 realify pass |
| `build_spdmx.py` | Post-render chunking of flat `SPDMX_dev/` into download chunks + `songs.csv` |
| `build_songs_table.py` | Aggregate `stems.csv` → `songs.csv` (`subset:all`, `subset:bdgp`, …) |
| `distribute_spdmx.py` | Stage Zenodo upload files (CSVs + per-chunk zips) |
| `spdmx_release/README.md` | **Public dataset schema** (stems/songs/chunks joins) — ships on Zenodo |
| `chunking.py` | Song→chunk assignment (~25 GiB) and packaged CSV helpers |
| `audio.py` | fluidsynth rendering, mono downmix, BS.1770-4 loudness, FLAC I/O, mixture build |
| `velocity.py` | MIDI max-velocity → per-track dynamics scales for mix |
| `mix.py` | Dataset-level stem normalization (summability) |
| `dataset.py` | Ablation sampling vs full-dataset filtering |
| `paths.py` | Output path helpers (`dev/ablations/`, `dev/stems/`, `dev/analysis/`, `SPDMX/`) |
| `patches.py` | Slakh-style patch randomization (stub) |
| `cli_common.py` | Shared argparse flags for synthesis CLIs |
| `listening/` | Localhost ablation comparison viewer (`python -m synthesis.listening.serve`) |
| `realify/` | Stable Audio 3 audio-to-audio wrapper and captions |
| `tests/` | Synthesis unit tests |

## Output layout (development)

On deepfreeze:

```
{OUTPUT_DIR}/dev/
├── ablations/{basic,basic_realify,slakh,slakh_realify,ddsp_basic,ddsp_basic_realify,ddsp_slakh,ddsp_slakh_realify,clips}/
├── stems/              # synthesize --full
└── stems_realify/
```

Browsable in-repo via gitignored symlink [`ablations_output/`](ablations_output/) (created by `shared.setup_symlinks` or `synthesize`).

Flat production render: `{OUTPUT_DIR}/SPDMX_dev/` via `synthesis.final`.
Chunked release: `{OUTPUT_DIR}/SPDMX/` via `build_spdmx`.

Post-render packaging for Zenodo:

```bash
# 1. Build chunked release in a NEW dir (flat SPDMX/ untouched; hardlinks by default)
uv run python -m synthesis.build_spdmx -o "$SPDMX_OUTPUT_DIR"
# → {OUTPUT}/SPDMX/chunk_N/ + stems.csv + songs.csv + chunks.csv

# 2. Stage loose metadata + chunk_N.zip for upload
uv run python -m synthesis.distribute_spdmx -o "$SPDMX_OUTPUT_DIR" \
  --stage-dir /path/to/zenodo_stage
```
