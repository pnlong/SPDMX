# sPDMX

Audio stems, full-song mixes, and dense MIDI for
[PDMX](https://zenodo.org/records/13763756) (Long et al., ICASSP 2025). Each
PDMX song is identified by **`song_id`**, the hashed layout path without
`./data/` or `.json` (for example
`8/44/QmQt11bci266XxFpztpSJmfH1ijHX8zRcmaVe1Y7akYZTZ`).

## Layout

Released / Zenodo layout (after packaging):

```
.
├── LICENSE
├── README.md
├── stems.csv          # stem-level (one row per rendered track)
├── songs.csv          # song-level (one row per song_id; subsets)
├── chunks.csv         # per-chunk size / archive metadata
├── chunk_0/
│   └── <song_id>/
│       ├── 0.flac
│       ├── 1.flac
│       ├── …
│       ├── mix.flac   # full-song mix (ffmpeg sum of stems; mono)
│       └── mix.mid    # dense MIDI
├── chunk_1/
│   └── <song_id>/…
└── ...
```

Lab / production flat tree (`SPDMX_dev/`) keeps separate trees:

```
SPDMX_dev/
├── audio/<song_id>/<track>.flac
├── mid/<song_id>.mid
└── mix/<song_id>.flac
```

`track` in filenames and in `stems.csv` is the **dense** MIDI track index
after empty PDMX tracks are dropped. `original_track` is the PDMX MIDI track
index.

Mixes are a **raw linear sum** of summable stems (`ffmpeg amix … normalize=0`),
written as **mono** FLAC to match the stem channel layout. Stereo models (e.g.
Stable Audio Open fine-tunes) should duplicate mono→L/R at load time rather than
storing stereo mixes in the release.

## Tables and joins

```
songs.csv  1 ──<  stems.csv     join key: song_id
   │                  │
   │                  └── path → ./chunk_N/{song_id}/   (+ {track}.flac)
   │                  └── mid  → ./chunk_N/{song_id}/mix.mid
   │                  └── mix  → ./chunk_N/{song_id}/mix.flac
   └── subset:bdgp / subset:all  (PDMX-style boolean filters)
```

| Table | Granularity | Primary key | Typical use |
|-------|-------------|-------------|-------------|
| **`stems.csv`** | one row per stem | `(song_id, track)` | load audio; instrument filters |
| **`songs.csv`** | one row per song | `song_id` | corpus subsets; song-level filters |
| **`chunks.csv`** | one row per chunk | `chunk` | Zenodo download planning |

Always prefer filtering **songs** first, then restricting stems:

```python
songs = pd.read_csv("songs.csv")
stems = pd.read_csv("stems.csv")

bdgp_ids = songs.loc[songs["subset:bdgp"], "song_id"]
stems_bdgp = stems[stems["song_id"].isin(bdgp_ids)]

# Or: all songs in a downloaded chunk
stems_c0 = stems[stems["chunk"] == 0]
```

### `stems.csv` columns

| Column | Type | Meaning |
|--------|------|---------|
| `song_id` | str | Song key; joins to `songs.csv` and (via strip) to PDMX |
| `path` | str | Release-relative song **directory** (`./chunk_N/{song_id}`) |
| `mid` | str | Release-relative dense MIDI (`./chunk_N/{song_id}/mix.mid`) |
| `mix` | str | Release-relative full mix (`./chunk_N/{song_id}/mix.flac`) |
| `track` | int | Dense track index (= `{track}.flac` basename) |
| `original_track` | int | Track index in the source PDMX MIDI (before dropping empties) |
| `program` | int | GM program number (0–127) |
| `is_drum` | bool | Drum / percussion channel |
| `name` | str | Optional track name from MIDI (may be empty) |
| `chunk` | int | Chunk id; media lives under `chunk_{chunk}/` |

Audio file for a row: `{path}/{track}.flac`. Full mix: `{mix}` (or
`{path}/mix.flac`). Dense MIDI: `{mid}` (or `{path}/mix.mid`).

### `songs.csv` columns

Built from `stems.csv` by `synthesis.build_songs_table` (also run at the end of
`synthesis.build_spdmx`). Pipe-delimited list fields use `|` (not commas).

| Column | Type | Meaning |
|--------|------|---------|
| `song_id` | str | Primary key; joins to `stems.csv.song_id` |
| `path` | str | Same as stem rows for this song |
| `mid` | str | Same as stem rows for this song |
| `mix` | str | Same as stem rows for this song |
| `chunk` | int | Same as stem rows (one chunk per song) |
| `n_tracks` | int | Number of stem rows in `stems.csv` for this song |
| `n_stems_on_disk` | int | Rows whose FLAC resolved at songs-table build time |
| `tracks` | str | Pipe-delimited dense track indices (`0\|1\|2`) |
| `original_tracks` | str | Pipe-delimited PDMX original track indices |
| `programs` | str | Pipe-delimited **sorted unique** MIDI programs (`0\|24\|32`) |
| `gm_classes` | str | Pipe-delimited GM class names (`bass\|drums\|piano`) |
| `bdgp_targets` | str | Pipe-delimited BDGP targets present (`bass\|drums\|…`) |
| `subset:all` | bool | ≥1 stem on disk at build time |
| `subset:bdgp` | bool | **B**ass, **D**rums, **G**uitar, and **P**iano all present |

**`subset:bdgp`** uses the same GM→target map as the separation PoC
(bass / drums / guitar / piano). Source separation packs and the SAO
**matched** arm use this subset; SAO **full** uses `subset:all`.

Example program filter (piano program 0 and any guitar 24–31):

```python
songs[
    songs["programs"].str.contains(r"(^|\|)0(\||$)", regex=True)
    & songs["programs"].str.contains(r"(^|\|)2[4-9](\||$)|(^|\|)3[01](\||$)", regex=True)
]
```

Refresh without re-chunking:

```bash
uv run python -m synthesis.build_songs_table --spdmx-root /path/to/SPDMX
```

### `chunks.csv` columns

| Column | Meaning |
|--------|---------|
| `chunk` | Integer id → directory `chunk_{chunk}/` |
| `n_songs` / `n_stems` | Counts in that chunk |
| `bytes` | Total media bytes |
| `archive` / `sha256` | Filled by `distribute_spdmx` for Zenodo zips |

## Primary key (PDMX join)

`song_id` is the song primary key. It joins to PDMX.csv as:

| sPDMX | PDMX.csv |
|-------|----------|
| `song_id` | `path` with `./data/` prefix and `.json` suffix stripped |
| `mid` (`./chunk_N/{song_id}/mix.mid`) | `mid` is the PDMX MIDI path; sPDMX `mid` is dataset-relative |
| `path` (`./chunk_N/{song_id}`) | PDMX `path` is metadata JSON; sPDMX `path` is the song directory |

## Zenodo download

On Zenodo, metadata files and chunk archives are **separate downloads**:

1. Download `stems.csv`, `songs.csv`, `chunks.csv`, `LICENSE`, and `README.md`.
2. Use `songs.csv` subsets and/or `chunks.csv` (or filter `stems.csv` by `chunk`) to choose media.
3. Download only the `chunk_N.zip` files you need (default **64** roughly
   equal-sized chunks).
4. Unzip each archive next to the CSVs so paths like `./chunk_0/<song_id>/…` resolve.
5. Restrict work to local media with
   `stems.csv[stems.csv["chunk"] == 0]` (or a set of chunk ids).

## Rebuild (lab)

After summable stems exist under `SPDMX_dev/audio/`:

```bash
uv run python -m synthesis.final --only-pass song_mix -j 8
# or: uv run python -m synthesis.render_mixes -j 8
uv run python -m synthesis.build_spdmx -j 8
```

## Citation

Please cite PDMX and this work if you use sPDMX.

```
@inproceedings{long2024pdmx,
  author={Long, Phillip and Novack, Zachary and Berg-Kirkpatrick, Taylor and McAuley, Julian},
  booktitle={ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  title={{PDMX}: A Large-Scale Public Domain MusicXML Dataset for Symbolic Music Processing},
  year={2025},
  pages={1-5},
  doi={10.1109/ICASSP49660.2025.10890217}
}
```

## License

See `LICENSE` in this directory (CC BY 4.0, with PDMX public-domain scores and
optional Stability AI terms for SA3-realified stems).
