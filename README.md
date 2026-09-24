# spdmx

Turn the [PDMX](https://zenodo.org/records/13763756) symbolic music dataset into
audio stems via a FluidSynth + MIDI-DDSP / DDSP-Piano hybrid pipeline.

**Project page:** [pnlong.github.io/SPDMX](https://pnlong.github.io/SPDMX/) (sources in [`docs/`](docs/))

## Pipeline

1. **Synthesis setup** — `python -m analysis.prepare_synthesis` (GM register + dense corrected MIDIs; **required before any ablation**)
2. **Synthesis** — `python -m synthesis.synthesize` with `--render-mode {basic,slakh,ddsp_basic,ddsp_slakh}`
3. **Full dataset** — `python -m synthesis.final --only-pass {layout,fluidsynth,ddsp_piano,midi_ddsp,mix}` (FLAC under `{OUTPUT_DIR}/SPDMX_dev/`)
4. **Analysis** — duration stats, GM register, DDSP coverage

Optional generative eval: [`experiments/sao`](experiments/sao) (Stable Audio Open fine-tune PoC).

## Install

**→ Full step-by-step guide: [`SETUP.md`](SETUP.md)**

**→ Join an existing production job on a new GPU machine: [`synthesis/FINAL_SETUP.md`](synthesis/FINAL_SETUP.md)**

**→ ICASSP pilots (YourMT3 + StreamGen) on a Deep Freeze machine: [`experiments/COLLABORATOR_SETUP.md`](experiments/COLLABORATOR_SETUP.md)**

Quick start (synthesis + analysis):

```bash
cd ~/spdmx
uv sync --group dev
uv run python -c "import mido, synthesis.audio; print('spdmx ok')"
```

For neural DDSP (`ddsp_basic` / `ddsp_slakh`), follow **Track C** in [`SETUP.md`](SETUP.md).

## Usage

Default output root: `/deepfreeze/share/SPDMX` (`SPDMX_OUTPUT_DIR` in [`.env`](.env.example); imported as `OUTPUT_DIR` from [`shared/config.py`](shared/config.py)).

Development artifacts (ablations, analysis) live under `{OUTPUT_DIR}/dev/`. Production stems go to `{OUTPUT_DIR}/SPDMX_dev/` via `synthesis.final`.

### Ablation (four render modes)

Default behavior: random sample from `subset:rated_deduplicated` (N=100, seed=42).

```bash
# Step 0 — correct GM ids from track names (once; re-run after alias YAML edits)
uv run python -m analysis.prepare_synthesis --subset all_valid -j 8
# → {OUTPUT_DIR}/dev/analysis/instruments/all_valid/register.csv

# A1 / B1 — FluidSynth stems (loads register by default)
uv run python -m synthesis.synthesize --render-mode basic
uv run python -m synthesis.synthesize --render-mode slakh

# CA1 / CB1 — neural DDSP + soundfont fallback (requires Track C / .venv-ddsp)
uv run python -m synthesis.synthesize --render-mode ddsp_basic
uv run python -m synthesis.synthesize --render-mode ddsp_slakh
```

Output:

```
/deepfreeze/pnlong/SPDMX/dev/ablations/
├── basic/
├── slakh/
├── ddsp_basic/
└── ddsp_slakh/
```

### Full sPDMX (after listening test)

```bash
# Step 0 (if not already): register + dense corrected MIDIs
uv run python -m analysis.prepare_synthesis --subset all_valid -j 8

uv run python -m synthesis.final --only-pass layout
uv run python -m synthesis.final --only-pass fluidsynth -j 8
uv run python -m synthesis.final --only-pass ddsp_piano
uv run python -m synthesis.final --only-pass midi_ddsp
uv run python -m synthesis.final --only-pass verify
uv run python -m synthesis.final --only-pass mix
```

**Multi-machine GPU rendering:** pass `--shard-count N --shard-index k` on
`fluidsynth`, `ddsp_piano`, or `midi_ddsp` (one unique index per machine). See
[`synthesis/FINAL_SETUP.md`](synthesis/FINAL_SETUP.md).

Writes raw FLAC stems to `{OUTPUT_DIR}/SPDMX_dev/raw/` (mix writes summable stems to `audio/`). Sanitized MIDIs, `stems.csv`, `LICENSE`, and `README.md` come from `prepare_synthesis` / layout. Mix is `sum(stems)` (no `mixture.*`). Pipeline tables live under `{OUTPUT_DIR}/dev/final/` (`stems.fluidsynth.csv` etc. during render; `stems.csv` / `data.csv` after mix or `--only-pass merge`).

### Per-song layout

```
{OUTPUT_DIR}/SPDMX_dev/
├── LICENSE
├── README.md
├── stems.csv                     # join to PDMX.csv on song_id; row key (song_id, track)
├── raw/<song_id>/                # pre-mix hybrid stems
│   ├── 0.flac
│   └── …
├── audio/<song_id>/              # mixable stems (after --only-pass mix)
│   ├── 0.flac
│   └── …
├── mid/<song_id>.mid
└── mix/<song_id>.flac            # full-song mix (written by --only-pass mix)
```

Chunked **release** tree `{OUTPUT_DIR}/SPDMX/` uses flattened
`chunk_N/<song_id>/{k.flac,mix.flac,mix.mid}` plus `chunks.csv` and song-level
`songs.csv` (`subset:all` / `subset:bdgp`). Schema and joins:
[`synthesis/spdmx_release/README.md`](synthesis/spdmx_release/README.md).

Rebuild release after mixes exist:

```bash
uv run python -m synthesis.final --only-pass mix -j 8
uv run python -m synthesis.build_spdmx -j 8
```

### Analysis

**GM register (prerequisite for synthesis):** corrects mismatched GM program ids from MIDI track names:

```bash
uv run python -m analysis.prepare_synthesis --subset all_valid -j 8
# Re-print stats without re-parsing MIDI:
uv run python -m analysis.prepare_synthesis --from-csv .../register.csv --no-write-corrected-midi
```

Writes to `{OUTPUT_DIR}/dev/analysis/instruments/all_valid/`: `register.csv`, `register_corrections.csv`, `register_summary.json`, `register_report.txt`, `register_top_corrections.csv`.

Song-length analysis uses PDMX metadata (`song_length.seconds`) — no synthesis required:

```bash
uv run python -m analysis.analyze_song_lengths
```

Writes to `{OUTPUT_DIR}/dev/analysis/song_lengths/`:

- `song_length_histogram.png` — duration distribution
- `song_length_percentiles.png` — empirical CDF (percentile curve)
- `song_length_report.json` — stats and duration percentiles

Also symlinks in-repo dev output (both gitignored; run `uv run python -m shared.setup_symlinks` after clone):

- [`analysis/output/`](analysis/output/) → `{OUTPUT_DIR}/dev/analysis/`
- [`synthesis/ablations_output/`](synthesis/ablations_output/) → `{OUTPUT_DIR}/dev/ablations/`

## Repository layout

| Path | Purpose |
|------|---------|
| [`SETUP.md`](SETUP.md) | **Environment setup guide** (uv, fluidsynth, DDSP) |
| `synthesis/synthesize.py` | Main CLI: ablation sample (default) or `--full` PDMX |
| `synthesis/final.py` | Hybrid production render from `recipe.yaml` |
| `synthesis/build_spdmx.py` | Post-render build `{OUTPUT}/SPDMX/` from flat `SPDMX_dev/` |
| `synthesis/distribute_spdmx.py` | Stage Zenodo metadata + per-chunk zips |
| `analysis/` | Duration / GM / DDSP coverage analysis — see [`analysis/README.md`](analysis/README.md) |
| `experiments/sao/` | Optional SAO fine-tune generative eval |
| `.env` | Machine paths (`SPDMX_PDMX_FILEPATH`, `SPDMX_OUTPUT_DIR`, …); copy from `.env.example` |
| `shared/config.py` | Constants + path imports from `.env` — see [`shared/README.md`](shared/README.md) |
| `shared/setup_symlinks.py` | Create in-repo symlinks after clone (`python -m shared.setup_symlinks`) |

See [`synthesis/RENDERING_NOTES.md`](synthesis/RENDERING_NOTES.md) for Slakh alignment, ablation design, and listening test plans. Synthesis layout: [`synthesis/README.md`](synthesis/README.md).

## Tests

```bash
uv run PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -p pytest
```

## License

See [LICENSE](LICENSE).
