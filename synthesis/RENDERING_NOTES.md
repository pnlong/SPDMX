# Rendering notes

## Output layout

Default root: `/deepfreeze/pnlong/SPDMX` (`OUTPUT_DIR` in [`shared/config.py`](../shared/config.py)).

Development artifacts live under `{OUTPUT_DIR}/dev/`. Flat production render is `{OUTPUT_DIR}/SPDMX_dev/`; chunked release is `{OUTPUT_DIR}/SPDMX/`.

**Ablation** (listening test; default `synthesize` behavior):

```
{OUTPUT_DIR}/dev/ablations/
├── listening_sample.yaml   # shared stratified song/stem inventory
├── basic/                  # A1
├── slakh/                  # B1
├── ddsp_basic/             # CA1 (neural DDSP + basic soundfont fallback copies)
├── ddsp_slakh/             # CB1 (neural DDSP + slakh soundfont fallback copies)
└── clips/{condition}/      # aligned 10s MP3 clips for listening.serve
```

If an older `slakh_ddsp/` tree exists, rename it to `ddsp_slakh/`.

**Production sPDMX** (`python -m synthesis.final`; default `--full`):

```
{OUTPUT_DIR}/SPDMX_dev/
├── LICENSE
├── README.md
├── stems.csv                     # PDMX song_id (./data/{song_id}.json)
├── audio/<song_id>/             # FLAC stems (0.flac, 1.flac, …)
│   ├── 0.flac
│   └── …
└── mid/<song_id>.mid            # sanitized dense MIDI
                                 # written by analysis.prepare_synthesis
```

Production bookkeeping (`data.csv`, `stems.csv`, `stem_recipe.csv`, plus per-pass `stems.<engine>.csv`) is `{OUTPUT_DIR}/dev/final/`, not the released tree.

Ablation-style full stems (`synthesize --full`) still use `{OUTPUT_DIR}/dev/stems/`.

**Analysis** (song lengths, GM register, etc.):

```
{OUTPUT_DIR}/dev/analysis/song_lengths/
{OUTPUT_DIR}/dev/analysis/instruments/all_valid/   # register.csv (step 0)
```

Output symlinked in-repo at [`analysis/output/`](../analysis/output/) → `{OUTPUT_DIR}/dev/analysis/` (gitignored).

Ablation outputs symlinked at [`synthesis/ablations_output/`](../synthesis/ablations_output/) → `{OUTPUT_DIR}/dev/ablations/` (gitignored).

Create both after clone: `uv run python -m shared.setup_symlinks`

**Assembled sPDMX dataset** (flat render via `synthesis.final`; chunk with `build_spdmx`):

```
{OUTPUT_DIR}/SPDMX_dev/
```

## GM register (step 0)

Before any ablation or `--full` run, build the per-track GM correction table:

```bash
uv run python -m analysis.prepare_synthesis --subset all_valid -j 8
```

- **Aliases:** [`analysis/gm_register_aliases.yaml`](../analysis/gm_register_aliases.yaml) (SATB→choir, harpsichord, sax vs alto, …)
- **Output:** `{OUTPUT_DIR}/dev/analysis/instruments/all_valid/register.csv` (+ corrections CSV, summary JSON, top-corrections CSV)
- **Re-run** after editing the alias YAML; then re-synthesize affected ablations with `--reset` if needed
- **Synthesize** loads that path by default (`--register` / `--no-register` to override)

Then run A1/B1/… as usual.

## Per-song layout

```
data/<mirrored-song-path>/
├── 0.mp3    # or 0.flac with --flac
├── 1.mp3
└── ...      # mix = sum(stems); no mixture.* on disk
```

Default on-disk format is **MP3**. Pass `--flac` to write FLAC stems (PCM_16). Use the same `--flac` flag for mix so they read and write the matching format.

## Mixture procedure

Canonical description (equations, motivation, constants): **[`MIXING.md`](MIXING.md)**.

Constant across all ablations (basic / slakh / ddsp_*):

| Setting | Value |
|---|---|
| Sample rate | 44.1 kHz |
| Stem channels | `STEM_CHANNELS` in `shared/config.py` (default `1` mono; `2` keeps fluidsynth stereo) |
| Loudness | Applied in `synthesis.mix` (−23 LUFS BS.1770-4, peak-limited to 1.0) |

1. Load raw stems; loudness-normalize toward −23 LUFS (BS.1770) with per-stem peak limiting at 1.0, then pad to equal length.
2. Multiply each stem by MIDI velocity dynamics \(s_i = v_i^{\max} / v_{\mathrm{song}}^{\max}\) (note-ons with velocity \(> 0\)).
3. Sum stems sample-wise (in memory).
4. If mixture peak > `MIXTURE_PEAK_LIMIT` (1.0), apply uniform gain `limit / peak` to every stem (same factor), so released stems remain linearly summable.
5. Overwrite `N.mp3` / `N.flac` with the scaled waveforms. **No `mixture.*` is written by default** — the mix is just `sum(stems)` (`--write-mixture` to also write it).

**Synthesis writes raw stems** (no LUFS). Summability normalization is a separate pass:

```bash
uv run python -m synthesis.mix --stems-dir /path/to/ablation -j 8
# or:
uv run python -m synthesis.mix --render-mode basic -j 8
# Preview without overwrite + write mixtures:
uv run python -m synthesis.mix --render-mode basic --no-overwrite --write-mixture -j 8
```

Implemented in [`audio.py`](audio.py), [`velocity.py`](velocity.py), [`mix.py`](mix.py).

## Commands

All synthesis flows go through `synthesis.synthesize` (expects GM `register.csv` unless `--no-register`):

```bash
COMMON="--sample-seed 43 -j 8"

# Step 0
python -m analysis.prepare_synthesis --subset all_valid -j 8

# FluidSynth (stratified sample written on first run → listening_sample.yaml)
python -m synthesis.synthesize --render-mode basic $COMMON          # A1
python -m synthesis.synthesize --render-mode slakh $COMMON          # B1

# DDSP (copies soundfont fallbacks from donors; renders neural stems only)
python -m synthesis.synthesize --render-mode ddsp_basic $COMMON     # CA1
python -m synthesis.synthesize --render-mode ddsp_slakh $COMMON     # CB1

# Optional: peak-normalize stems so mix = sum(stems)
python -m synthesis.mix --render-mode basic -j 8

# Aligned 10s clips (windows from A1) + listening viewer
python -m synthesis.listening.make_clips --clip-seconds 10
python -m synthesis.listening.serve

# Full PDMX after listening test (dense corrected MIDI)
python -m analysis.prepare_synthesis --subset all_valid -j 8
python -m synthesis.synthesize --render-mode basic --full
python -m synthesis.mix --full -j 8
```

## Hybrid final synthesis (per-category recipe)

After the ablation listening test, edit [`recipe.yaml`](recipe.yaml) with the winning ablation id per listening category (`basic`, `slakh`, `ddsp_basic`, `ddsp_slakh`).

`python -m synthesis.final --only-pass …` renders **one** mixed stem tree (not eight ablation dirs). Passes are **one method at a time** (do not omit `--only-pass`):

0. **layout** — mkdir `{OUTPUT_DIR}/SPDMX_dev/audio/<shard>/<hash>/` for every valid PDMX song, plus `mid/` parent dirs. Empty `data.csv` / `stems.csv` / `stem_recipe.csv` and `midi_index.csv` (dense MIDI path + `n_tracks` per song) under `{OUTPUT_DIR}/dev/final/`.
1. **Fluidsynth** — categories whose recipe is `basic` / `slakh` (and DDSP-ineligible fallbacks). Per-track slakh recipes vs default GM. `-j` workers. Progress is `stems.fluidsynth.csv` / `stem_recipe.fluidsynth.csv` (append-only).
2. **DDSP-Piano** — only if the **piano** category recipe is `ddsp_*` (acoustic-piano engine). Current `recipe.yaml` uses slakh for piano, so this pass is omitted. Writes `stems.ddsp_piano.csv` / `stem_recipe.ddsp_piano.csv`.
3. **MIDI-DDSP** — strings/wind/brass whose recipe is `ddsp_*`. May run **in parallel** with Fluidsynth and with DDSP-Piano when that pass exists. Writes `stems.midi_ddsp.csv` / `stem_recipe.midi_ddsp.csv`.
4. **merge** — optional. Concatenates the per-pass CSVs into canonical `stems.csv` / `stem_recipe.csv` / `ddsp_routing.csv` and writes `data.csv` for songs whose stem count matches `midi_index.csv`. Mix runs this automatically. **Pass shards stay on disk** so a later re-render or recipe change can still append to them.
5. **mix** — merge tables, then LUFS + velocity + peak into `audio/` **and** write `mix/<song_id>.flac` (ffmpeg stem sum) in the same pass. Always runs claimed-stem checks. Dirty-aware: re-normalize when any raw stem is newer than `audio/`; remake song mixes when any `audio/` stem is newer than the mix (or the song was just re-normalized); sum-checks only touched songs. `--verify` adds a full-catalog decode+mix=sum (same as the verify pass). `--repair-mix-sums` full-scans, deletes bad mix+audio, rebuilds from `raw/`, and covers verify.
6. **verify** — optional standalone gate: claimed stems + raw completeness + full decode+mix=sum. Prefer `mix --verify` or `mix --repair-mix-sums` when you are mixing anyway.

Without `--reset`, each method pass **resumes** from its own `stem_recipe.<pass>.csv` plus a valid on-disk FLAC (default; `--no-resume-check-disk` for CSV-only). Canonical `stems.csv` / `stem_recipe.csv` / `data.csv` are merge outputs (rebuilt at mix) and are deleted when a render pass starts; the per-pass shards are not. Valid stems whose pass sidecar matches the current recipe are skipped. Pass `-y` / `--yes` after a recipe change to regenerate mismatches without a prompt.

```bash
# Edit synthesis/recipe.yaml, then one pass per job (FLAC default):
uv run python -m synthesis.final --only-pass layout
uv run python -m synthesis.final --only-pass fluidsynth -j 8
uv run python -m synthesis.final --only-pass ddsp_piano
uv run python -m synthesis.final --only-pass midi_ddsp
uv run python -m synthesis.final --only-pass mix -j 8
uv run python -m synthesis.final --only-pass verify -j 8

# Optional: rebuild canonical CSVs without mixing (mix/verify already do this):
uv run python -m synthesis.final --only-pass merge

# Recipe changed; preview conflicts without writing, then regenerate:
uv run python -m synthesis.final --only-pass fluidsynth --dry-run
uv run python -m synthesis.final --only-pass fluidsynth -y -j 8
# Then mix (only dirty songs) + verify:
uv run python -m synthesis.final --only-pass mix -j 8
uv run python -m synthesis.final --only-pass verify -j 8

# Stratified sample instead of all valid PDMX (writes dev/ablations/final/):
uv run python -m synthesis.final --only-pass layout --ablation-sample
```

`--only-pass` is required. `--full` is the default. Stems are always FLAC (`N.flac`). MIDI + `stems.csv` + `LICENSE` + `README.md`: `{OUTPUT_DIR}/SPDMX_dev/` from `prepare_synthesis` (`song_id` joins to PDMX.csv; `path`/`mid` are dataset-relative). Tables: `{OUTPUT_DIR}/dev/final/`. Do not add `final` to listening `CONDITION_ORDER`. Render passes append `stem_recipe.<engine>.csv`; mix merges those into `stem_recipe.csv` beside `stems.csv` (`path`, `track`, `category`, `ablation`, `method`, `fallback`, `backend`).

Synthesize always uses those dense corrected MIDIs (`prepare_synthesis` is the step-0 setup).

Song-length analysis (no synthesis required):

```bash
python -m analysis.analyze_song_lengths
```

Neural-DDSP coverage (for the paper / sampling design):

```bash
python -m analysis.ddsp_coverage --subset rated_deduplicated
python -m analysis.ddsp_coverage --subset rated_deduplicated --check-monophony -n 500
```

Production layout is written by `python -m synthesis.final` (flat `audio/` + `mid/`). After the full render, `python -m synthesis.build_spdmx` builds a **new** `{OUTPUT}/SPDMX/` tree (hardlinks; flat `SPDMX_dev/` untouched), then stage zips with `python -m synthesis.distribute_spdmx --stage-dir …`.

## Module layout

```
synthesis/
├── synthesize.py       # ablation CLI (--render-mode, --full)
├── final.py            # hybrid production CLI (per-category recipe.yaml)
├── recipe.yaml         # per-category synthesis recipe (edit this)
├── recipe.py           # parse recipe → per-track plan + stem_recipe.csv
├── build_spdmx.py      # post-render: SPDMX_dev/ → SPDMX/chunk_N/
├── distribute_spdmx.py # stage Zenodo metadata + chunk_NNN.zip
├── chunking.py         # song→chunk assignment helpers
├── ddsp/               # MIDI-DDSP + DDSP-Piano workers
```

## Ablation study

| ID | Flags | Output |
|----|-------|--------|
| A1 | `basic` | `dev/ablations/basic/` |
| B1 | `slakh` | `dev/ablations/slakh/` |
| CA1 | `ddsp_basic` | `dev/ablations/ddsp_basic/` |
| CB1 | `ddsp_slakh` | `dev/ablations/ddsp_slakh/` |

Shared stratified sample (`listening_sample.yaml`, seed 43, ≥50 stems/category) ensures all conditions render the same songs.

**Donor reuse (NFS-safe copies):** CA/CB soundfont-fallback stems are `copy2`'d from A/B. Neural stems are newly rendered. Provenance is in `ddsp_routing.csv`:

| Column | Meaning |
|--------|---------|
| `path` | Song directory in this ablation (same as `stems.csv`) |
| `source` | `rendered` or `reused:basic` / `reused:slakh` |
| `original_path` | Absolute stem filepath copied from; `NA` when newly rendered |

No symlinks/hardlinks.

### Slakh mode (`--render-mode slakh`)

Slakh-style rendering adds **per-track patch variety** on top of basic Fluidsynth:

- Each listening category (piano, strings, wind, …) can use a different soundfont, FX profile, and GM program pool (from patch sweep tuning → `winners_locked.yaml`).
- Within a song, each track randomly draws a program from its category's pool (`select_patch` in [`patches.py`](patches.py)). Tracks sharing the same GM instrument class in a song get the **same** patch; the draw varies across songs (seeded by `(sample_seed, song_path, gm_class)`).
- Pools are defined in `PATCH_POOLS` (`pool_v1_conservative`, `pool_v2_diverse`, `pool_v3_slakh_like`). Until winners are locked, slakh mode passes MIDI programs through unchanged (same as basic).

See [`experiments/TUNING.md`](../experiments/TUNING.md) for the phased tuning workflow (soundfonts → FX → pools).

### Neural DDSP modes (`ddsp_basic` / `ddsp_slakh`)

Hybrid per-stem backends. Soundfont fallbacks copy from **basic** (`ddsp_basic`) or **slakh** (`ddsp_slakh`):

| Stem | Backend |
|------|---------|
| Acoustic hammer piano (GM **0, 1, 3**) / acoustic piano names | **DDSP-Piano** (MAESTRO; polyphony OK) |
| Harpsichord, clavinet, e-piano, electric grand (GM 2, 4–7) | donor soundfont copy |
| 13 URMP instruments, **monophonic** | **MIDI-DDSP** |
| Timbre mismatches (piccolo, pan flute, english horn, muted trumpet, …) | donor soundfont copy |
| Polyphonic URMP-eligible stems | donor soundfont copy |
| Drums, guitar, bass guitar, vocals, synths, other | donor soundfont copy |

Routing details live in [`synthesis/ddsp/routing.py`](ddsp/routing.py) (`DDSP_PIANO_PROGRAMS`, name deny-lists, SATB-vs-sax vocal guard).

- Neural models run in an isolated TF venv (`.venv-ddsp`); see SETUP Track C. Linux x86_64 only.
- **Persistent multi-GPU pool** (default): one long-lived `worker serve` process per id in `CUDA_VISIBLE_DEVICES`. Hybrid MIDI-DDSP / DDSP-Piano keep **one song thread per GPU** so tracks from different songs fill idle cards (not only stems inside one song). Ablation DDSP still uses song-level `-j 1` (spawn). `SPDMX_DDSP_ONESHOT=1` = legacy per-stem subprocesses; `SPDMX_DDSP_FORCE_CPU=1` → CPU worker.
- Routing decisions are written to `ddsp_routing.csv` beside the ablation tables.
- Provenance: [`THIRD_PARTY.md`](../THIRD_PARTY.md). Vocals deliberately stay on soundfont; lyric SVS is out of scope.

## Listening test

Stem-level comparison across A1/B1/CA1/CB1. After ablations, build aligned **10s** clips (windows chosen from A1) and serve the clips tree:

```bash
uv run python -m synthesis.listening.make_clips --clip-seconds 10
uv run python -m synthesis.listening.serve
```

See [`listening/README.md`](listening/README.md).

## Status

| Feature | Status |
|---------|--------|
| Mono + BS.1770 stems | Done |
| `--render-mode` on synthesize | Done |
| `--render-mode ddsp_basic` / `ddsp_slakh` + donor copy reuse | Done (isolated TF venv; SETUP Track C) |
| Stratified listening sample + 10s clips | Done |
| `--full` for all valid PDMX | Done |
| Hybrid `synthesis.final` from `recipe.yaml` | Done (FLAC in `{OUTPUT_DIR}/SPDMX_dev/`, mix = sum) |
| `build_spdmx.py` / `distribute_spdmx.py` | Done (post-render chunk + Zenodo staging) |
| Patch pools (Slakh) | Stub |
| `mixture` per song | Not stored; `synthesis.final` (and `synthesis.mix`) apply LUFS × velocity × peak so mix = sum(stems). See [`MIXING.md`](MIXING.md). |
| Listening test | Viewer available (`python -m synthesis.listening.serve`) |
| Song-length analysis (PDMX metadata + plots) | Done |
| Neural-DDSP coverage (`analysis.ddsp_coverage`) | Done |
