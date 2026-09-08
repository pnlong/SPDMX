# sPDMX

Audio stems and dense MIDI for [PDMX](https://zenodo.org/records/13763756)
(Long et al., ICASSP 2025). Each PDMX song is identified by **`song_id`**,
the hashed layout path without `./data/` or `.json`
(for example `8/44/QmQt11bci266XxFpztpSJmfH1ijHX8zRcmaVe1Y7akYZTZ`).

## Layout

Released / Zenodo layout (after packaging):

```
.
├── LICENSE
├── README.md
├── SPDMX.csv
├── chunks.csv
├── chunk_0/
│   ├── audio/<song_id>/<track>.flac
│   └── mid/<song_id>.mid
├── chunk_1/
│   ├── audio/...
│   └── mid/...
└── ...
```

`track` in filenames and in `SPDMX.csv` is the **dense** MIDI track index
after empty PDMX tracks are dropped. `original_track` is the PDMX MIDI track
index.

## Primary key

`song_id` is the song primary key. It joins to PDMX.csv as:

| sPDMX | PDMX.csv |
|-------|----------|
| `song_id` | `path` with `./data/` prefix and `.json` suffix stripped |
| `mid` (`./chunk_N/mid/{song_id}.mid`) | `mid` is the PDMX MIDI path; sPDMX `mid` is dataset-relative |
| `path` (`./chunk_N/audio/{song_id}`) | PDMX `path` is metadata JSON; sPDMX `path` is the stem directory |

Row identity in `SPDMX.csv` is `(song_id, track)`.

Columns: `song_id`, `path`, `mid`, `track`, `original_track`, `program`,
`is_drum`, `name`, `chunk`.

The **`chunk`** column is an integer id (`0`, `1`, `2`, …) naming the
`chunk_N/` directory that holds that song’s audio and MIDI. All stems for a
song share one chunk.

## Zenodo download

On Zenodo, metadata files and chunk archives are **separate downloads**:

1. Download `SPDMX.csv`, `chunks.csv`, `LICENSE`, and `README.md`.
2. Use `chunks.csv` (or filter `SPDMX.csv` by `chunk`) to choose media.
3. Download only the `chunk_N.zip` files you need (~25 GB each).
4. Unzip each archive next to the CSVs so paths like
   `./chunk_0/audio/...` resolve.
5. Restrict work to local media with
   `SPDMX.csv[SPDMX.csv["chunk"] == 0]` (or a set of chunk ids).

`chunks.csv` columns: `chunk`, `n_songs`, `n_stems`, `bytes`, `archive`,
`sha256`.

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
