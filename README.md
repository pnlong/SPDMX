# spdmx

Turn the [PDMX](https://zenodo.org/records/13763756) symbolic music dataset into audio stems, captions, and SA3-realified audio.

**Project page:** [pnlong.github.io/SPDMX](https://pnlong.github.io/SPDMX/) (sources in [`docs/`](docs/))

## Pipeline

1. **Synthesis setup** — `python -m analysis.prepare_synthesis` (GM register + dense corrected MIDIs; **required before any ablation**)
2. **Synthesis** — `python -m synthesis.synthesize` with `--render-mode {basic,slakh,ddsp_basic,ddsp_slakh}`
3. **Realify** (optional) — same command with `--realify`
4. **Full dataset** — `python -m synthesis.final --only-pass {layout,fluidsynth,ddsp_piano,midi_ddsp,mix}` (FLAC under `{OUTPUT_DIR}/SPDMX_dev/`)
5. **Analysis** — duration stats and SA3 model recommendation

## Install

**→ Full step-by-step guide: [`SETUP.md`](SETUP.md)**

**→ Join an existing production job on a new GPU machine: [`synthesis/FINAL_SETUP.md`](synthesis/FINAL_SETUP.md)**

Quick start (synthesis + analysis only):

```bash
cd ~/spdmx
uv sync --group dev
uv run python -c "import mido, synthesis.audio; print('spdmx ok')"
```

For SA3 realify, submodule, flash-attention, and Hugging Face login, follow **Track B** in [`SETUP.md`](SETUP.md).

## Usage

Default output root: `/deepfreeze/share/SPDMX` (`SPDMX_OUTPUT_DIR` in [`.env`](.env.example); imported as `OUTPUT_DIR` from [`shared/config.py`](shared/config.py)).

Development artifacts (ablations, analysis) live under `{OUTPUT_DIR}/dev/`. Production stems go to `{OUTPUT_DIR}/SPDMX_dev/` via `synthesis.final`.

### Ablation (four conditions)

Default behavior: random sample from `subset:rated_deduplicated` (N=100, seed=42).

```bash
# Step 0 — correct GM ids from track names (once; re-run after alias YAML edits)
uv run python -m analysis.prepare_synthesis --subset all_valid -j 8
# → {OUTPUT_DIR}/dev/analysis/instruments/all_valid/register.csv

# A1 / B1 — raw stems (loads register by default)
uv run python -m synthesis.synthesize --render-mode basic
uv run python -m synthesis.synthesize --render-mode slakh

# Prototyping: MP3 instead of FLAC (smaller; use same flag for realify)
uv run python -m synthesis.synthesize --render-mode basic
uv run python -m synthesis.synthesize --render-mode basic --realify

# A2 / B2 — realify (GPU only; requires A1 / B1 stems first)
uv run python -m synthesis.synthesize --render-mode basic --realify
uv run python -m synthesis.synthesize --render-mode slakh --realify
```

Output:

```
/deepfreeze/pnlong/SPDMX/dev/ablations/
├── basic/
├── basic_realify/
├── slakh/
└── slakh_realify/
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
`fluidsynth`, `ddsp_piano`, `midi_ddsp`, or `realify` (one unique index per
machine). See [`synthesis/FINAL_SETUP.md`](synthesis/FINAL_SETUP.md).

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
└── mix/<song_id>.flac            # full-song mix (after --only-pass song_mix)
```

Chunked **release** tree `{OUTPUT_DIR}/SPDMX/` uses flattened
`chunk_N/<song_id>/{k.flac,mix.flac,mix.mid}` plus `chunks.csv` and song-level
`songs.csv` (`subset:all` / `subset:bdgp`). Schema and joins:
[`synthesis/spdmx_release/README.md`](synthesis/spdmx_release/README.md).

Rebuild release after mixes exist::

```bash
uv run python -m synthesis.final --only-pass song_mix -j 8
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

- `song_length_histogram.png` — distribution with SA3 limits marked
- `song_length_percentiles.png` — empirical CDF (percentile curve)
- `song_length_report.json` — stats, duration percentiles, SA3 limit percentiles, and model recommendation

Also symlinks in-repo dev output (both gitignored; run `uv run python -m shared.setup_symlinks` after clone):

- [`analysis/output/`](analysis/output/) → `{OUTPUT_DIR}/dev/analysis/`
- [`synthesis/ablations_output/`](synthesis/ablations_output/) → `{OUTPUT_DIR}/dev/ablations/`

## Repository layout

| Path | Purpose |
|------|---------|
| [`SETUP.md`](SETUP.md) | **Environment setup guide** (uv, SA3, flash-attn) |
| `synthesis/synthesize.py` | Main CLI: ablation sample (default) or `--full` PDMX |
| `synthesis/build_spdmx.py` | Post-render build `{OUTPUT}/SPDMX/` from flat `SPDMX_dev/` |
| `synthesis/distribute_spdmx.py` | Stage Zenodo metadata + per-chunk zips |
| `synthesis/realify/` | SA3 wrapper + submodule |
| `synthesis/realify/captions/` | Caption generation from PDMX metadata |
| `analysis/` | Duration analysis and SA3 model recommendation — see [`analysis/README.md`](analysis/README.md) |
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
