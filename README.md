# SPDMX

**SPDMX** turns [PDMX](https://zenodo.org/records/13763756) symbolic scores into open multitrack stem audio at corpus scale. MIDI is rendered with a FluidSynth + MIDI-DDSP / DDSP-Piano hybrid pipeline into mono FLAC stems, full-song mixes, and dense MIDI.

**Project page:** [pnlong.github.io/SPDMX](https://pnlong.github.io/SPDMX/) · sources in [`docs/`](docs/)

## Install

Full machine setup (uv, FluidSynth, optional neural DDSP): **[`SETUP.md`](SETUP.md)**

```bash
cp .env.example .env   # set PDMX / output / soundfont paths
uv sync --group dev
uv run python -c "import mido, synthesis.audio; print('spdmx ok')"
```

Neural DDSP backends (`ddsp_basic` / `ddsp_slakh`) need Track C in [`SETUP.md`](SETUP.md).

## Pipeline

1. **Prepare** — correct GM program IDs from track names and write dense MIDIs  
   `python -m analysis.prepare_synthesis`
2. **Ablate** (optional) — sample renders for listening / recipe tuning  
   `python -m synthesis.synthesize --render-mode {basic,slakh,ddsp_basic,ddsp_slakh}`
3. **Render** — hybrid production stems from [`synthesis/recipe.yaml`](synthesis/recipe.yaml)  
   `python -m synthesis.final --only-pass {layout,fluidsynth,ddsp_piano,midi_ddsp,mix,…}`
4. **Package** — chunked release tree for distribution  
   `python -m synthesis.build_spdmx`

Paths come from `.env` (`SPDMX_OUTPUT_DIR`, imported as `OUTPUT_DIR` in [`shared/config.py`](shared/config.py)). Dev artifacts live under `{OUTPUT_DIR}/dev/`; production stems under `{OUTPUT_DIR}/SPDMX_dev/`; the public chunked layout under `{OUTPUT_DIR}/SPDMX/`.

## Usage

### Prepare (required before synthesis)

```bash
uv run python -m analysis.prepare_synthesis --subset all_valid -j 8
```

### Ablation sample

Default: random sample from `subset:rated_deduplicated` (N=100, seed=42).

```bash
uv run python -m synthesis.synthesize --render-mode basic
uv run python -m synthesis.synthesize --render-mode slakh
# Neural DDSP (Track C):
uv run python -m synthesis.synthesize --render-mode ddsp_basic
uv run python -m synthesis.synthesize --render-mode ddsp_slakh
```

Outputs: `{OUTPUT_DIR}/dev/ablations/{basic,slakh,ddsp_basic,ddsp_slakh}/`

### Full corpus render

```bash
uv run python -m analysis.prepare_synthesis --subset all_valid -j 8
uv run python -m synthesis.final --only-pass layout
uv run python -m synthesis.final --only-pass fluidsynth -j 8
uv run python -m synthesis.final --only-pass ddsp_piano
uv run python -m synthesis.final --only-pass midi_ddsp
uv run python -m synthesis.final --only-pass verify
uv run python -m synthesis.final --only-pass mix
```

Shard across machines with `--shard-count N --shard-index k` on the render passes — see [`synthesis/FINAL_SETUP.md`](synthesis/FINAL_SETUP.md).

### Release layout

```
{OUTPUT_DIR}/SPDMX/
├── LICENSE
├── README.md
├── stems.csv / songs.csv / chunks.csv
└── chunk_N/<song_id>/{0.flac,…,mix.flac,mix.mid}
```

Flat working tree during render:

```
{OUTPUT_DIR}/SPDMX_dev/
├── raw/<song_id>/     # pre-mix hybrid stems
├── audio/<song_id>/   # summable stems (after mix)
├── mid/<song_id>.mid
└── mix/<song_id>.flac
```

Schema and joins: [`synthesis/spdmx_release/README.md`](synthesis/spdmx_release/README.md). Rebuild the chunked tree after mixes exist:

```bash
uv run python -m synthesis.final --only-pass mix -j 8
uv run python -m synthesis.build_spdmx -j 8
```

### Analysis

```bash
uv run python -m analysis.analyze_song_lengths
```

More: [`analysis/README.md`](analysis/README.md). After clone, optional in-repo output symlinks:

```bash
uv run python -m shared.setup_symlinks
```

## Repository

| Path | Purpose |
|------|---------|
| [`SETUP.md`](SETUP.md) | Environment setup |
| [`synthesis/`](synthesis/) | Rendering, mixing, packaging — see [`synthesis/README.md`](synthesis/README.md) |
| [`analysis/`](analysis/) | GM register, durations, coverage |
| [`experiments/`](experiments/) | Listening, Demucs, SAO, transcription, StreamGen |
| [`docs/`](docs/) | Project page (GitHub Pages) |
| [`shared/`](shared/) | Config and path helpers |
| `.env` | Machine paths (copy from [`.env.example`](.env.example)) |

Rendering notes and ablation design: [`synthesis/RENDERING_NOTES.md`](synthesis/RENDERING_NOTES.md). Third-party code: [`THIRD_PARTY.md`](THIRD_PARTY.md).

## Tests

```bash
uv run PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -p pytest
```

## License

[MIT](LICENSE)
