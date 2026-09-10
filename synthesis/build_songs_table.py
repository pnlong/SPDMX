"""Build a song-level sPDMX table with PDMX-style boolean subset columns.

``stems.csv`` is stem-level (``song_id``, ``track``, …). This writes
``songs.csv`` (one row per song) with:

- ``programs`` — pipe-delimited sorted unique MIDI program numbers (easy makeup filters)
- ``gm_classes`` — pipe-delimited GM class names (via program / drum flag)
- ``song_length`` — mix ``.flac`` duration in seconds (via ``soundfile.info``)
- ``subset:all`` — every song with ≥1 on-disk stem
- ``subset:bdgp`` — bass, drums, guitar, and piano all present (GM mapping)

Mirrors PDMX's ``subset:*`` columns on the songs table.

Example filter (songs with piano program 0 and any guitar 24–31)::

    songs[songs["programs"].str.contains(r"(^|\\|)0(\\||$)", regex=True)]
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from experiments.separation.audio_io import audio_duration_seconds
from experiments.separation.paths import SPDMX_ROOT, TARGETS
from experiments.separation.spdmx_io import resolve_spdmx_mix, resolve_spdmx_stem
from experiments.separation.targets import gm_to_target
from shared.config import (
    OUTPUT_DIR,
    SPDMX_BDGP_SUBSET_COLUMN,
    SPDMX_DATASET_DIR_NAME,
    SPDMX_FILE_NAME,
    SPDMX_SONGS_FILE_NAME,
)
from synthesis.patches import patch_group_key

# PDMX-style subset column names.
SUBSET_ALL = "subset:all"
SUBSET_BDGP = SPDMX_BDGP_SUBSET_COLUMN
# Prefer | over comma so naive string filters don't fight CSV quoting.
PROGRAMS_SEP = "|"
# Wall-clock seconds from the shipped mix FLAC header.
SONG_LENGTH_COLUMN = "song_length"

SONGS_FILE_NAME = SPDMX_SONGS_FILE_NAME


def _coerce_flags(df: pd.DataFrame) -> pd.DataFrame:
    if "chunk" in df.columns:
        df = df.copy()
        df["chunk"] = df["chunk"].map(lambda x: x if pd.isna(x) else str(int(x)))
    if "is_drum" in df.columns:
        df = df.copy()
        df["is_drum"] = df["is_drum"].astype(str).str.lower().isin(("true", "1", "yes"))
    return df


def mix_duration_seconds(mix_path: Path | None) -> float | None:
    """Return ``soundfile`` duration for a mix FLAC, or ``None`` if missing/unreadable."""
    if mix_path is None or not Path(mix_path).is_file():
        return None
    try:
        dur = float(audio_duration_seconds(Path(mix_path)))
    except Exception:  # noqa: BLE001
        return None
    return dur if dur > 0 else None


def build_songs_table(
    spdmx_root: Path,
    *,
    stems_csv: str = f"{SPDMX_FILE_NAME}.csv",
    check_files: bool = True,
) -> pd.DataFrame:
    """Aggregate stem rows into a song table with subset flags."""
    csv_path = spdmx_root / stems_csv
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    df = _coerce_flags(pd.read_csv(csv_path))
    rows: list[dict] = []
    grouped = df.groupby("song_id", sort=False)
    for song_id, g in tqdm(grouped, desc="songs:index", total=int(df["song_id"].nunique())):
        song_id_s = str(song_id)
        head = g.iloc[0]
        path = head.get("path")
        mid = head.get("mid")
        mix = head.get("mix") if "mix" in head.index else None
        chunk = head.get("chunk")
        present: set[str] = set()
        programs: set[int] = set()
        gm_classes: set[str] = set()
        tracks: set[int] = set()
        original_tracks: set[int] = set()
        n_stems_on_disk = 0
        for _, r in g.iterrows():
            if check_files:
                stem = resolve_spdmx_stem(
                    spdmx_root, r, song_id=song_id_s, track=int(r["track"]),
                )
                if stem is None:
                    continue
                n_stems_on_disk += 1
            else:
                n_stems_on_disk += 1
            prog = int(r["program"])
            is_drum = bool(r["is_drum"])
            tracks.add(int(r["track"]))
            if "original_track" in r.index and pd.notna(r["original_track"]):
                original_tracks.add(int(r["original_track"]))
            programs.add(prog)
            gm_classes.add(patch_group_key(prog, is_drum))
            t = gm_to_target(prog, is_drum)
            if t is not None:
                present.add(t)
        if n_stems_on_disk == 0:
            continue
        mix_path = resolve_spdmx_mix(spdmx_root, head, song_id=song_id_s)
        rows.append(
            {
                # Primary key: joins to stems.csv.song_id (stem-level table).
                "song_id": song_id_s,
                "path": path,
                "mid": mid,
                "mix": mix,
                "chunk": chunk,
                "n_tracks": int(len(g)),
                "n_stems_on_disk": int(n_stems_on_disk),
                SONG_LENGTH_COLUMN: mix_duration_seconds(mix_path),
                # Dense track indices in this release (join stems.csv on song_id+track).
                "tracks": PROGRAMS_SEP.join(str(t) for t in sorted(tracks)),
                "original_tracks": PROGRAMS_SEP.join(
                    str(t) for t in sorted(original_tracks)
                ),
                "programs": PROGRAMS_SEP.join(str(p) for p in sorted(programs)),
                "gm_classes": PROGRAMS_SEP.join(sorted(gm_classes)),
                "bdgp_targets": PROGRAMS_SEP.join(sorted(present)),
                SUBSET_ALL: True,
                SUBSET_BDGP: set(TARGETS).issubset(present),
            }
        )
    return pd.DataFrame(rows)


def write_songs_table(
    spdmx_root: Path,
    *,
    out_path: Path | None = None,
    check_files: bool = True,
) -> Path:
    songs = build_songs_table(spdmx_root, check_files=check_files)
    dest = out_path or (spdmx_root / SONGS_FILE_NAME)
    dest.parent.mkdir(parents=True, exist_ok=True)
    songs.to_csv(dest, index=False)
    n_len = int(songs[SONG_LENGTH_COLUMN].notna().sum()) if len(songs) else 0
    summary = {
        "path": str(dest),
        "n_songs": int(len(songs)),
        "n_with_song_length": n_len,
        "n_subset_all": int(songs[SUBSET_ALL].sum()) if len(songs) else 0,
        "n_subset_bdgp": int(songs[SUBSET_BDGP].sum()) if len(songs) else 0,
    }
    with open(dest.with_suffix(".summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return dest


def _duration_job(payload: tuple[str, str]) -> tuple[str, float | None]:
    """Picklable worker: ``(song_id, mix_path_str)`` → duration."""
    song_id, mix_path = payload
    return song_id, mix_duration_seconds(Path(mix_path) if mix_path else None)


def enrich_songs_csv_song_lengths(
    spdmx_root: Path,
    *,
    jobs: int = 8,
) -> Path:
    """Rewrite ``songs.csv`` ``song_length`` from on-disk mix FLAC headers."""
    root = Path(spdmx_root)
    dest = root / SONGS_FILE_NAME
    if not dest.is_file():
        raise FileNotFoundError(dest)
    songs = pd.read_csv(dest)
    payloads: list[tuple[str, str]] = []
    for _, row in songs.iterrows():
        sid = str(row["song_id"])
        mix = resolve_spdmx_mix(root, row, song_id=sid)
        payloads.append((sid, str(mix) if mix is not None else ""))

    lengths: dict[str, float | None] = {}
    n_jobs = max(1, int(jobs))
    if n_jobs == 1:
        for item in tqdm(payloads, desc="songs:mix-duration"):
            sid, dur = _duration_job(item)
            lengths[sid] = dur
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            futs = [pool.submit(_duration_job, item) for item in payloads]
            for fut in tqdm(
                as_completed(futs), total=len(futs), desc=f"songs:mix-duration (-j {n_jobs})"
            ):
                sid, dur = fut.result()
                lengths[sid] = dur

    songs[SONG_LENGTH_COLUMN] = songs["song_id"].astype(str).map(lengths)
    songs.to_csv(dest, index=False)
    return dest


def load_bdgp_song_ids(spdmx_root: Path) -> set[str] | None:
    """Return song_ids with ``subset:bdgp`` if songs.csv exists."""
    path = spdmx_root / SONGS_FILE_NAME
    if not path.is_file():
        return None
    df = pd.read_csv(path)
    if SUBSET_BDGP not in df.columns:
        return None
    flag = df[SUBSET_BDGP]
    if flag.dtype == object:
        flag = flag.astype(str).str.lower().isin(("true", "1", "yes"))
    return set(df.loc[flag, "song_id"].astype(str))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spdmx-root",
        type=Path,
        default=None,
        help=f"sPDMX release root (default: {{OUTPUT_DIR}}/{SPDMX_DATASET_DIR_NAME})",
    )
    parser.add_argument("--out", type=Path, default=None, help="songs.csv path")
    parser.add_argument(
        "--enrich-lengths-only",
        action="store_true",
        help="Only refresh song_length from mix FLAC headers (no stem scan)",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=8,
        help="Parallel workers for --enrich-lengths-only (default: 8)",
    )
    parser.add_argument(
        "--no-check-files",
        action="store_true",
        help="Trust CSV rows without verifying stem files on disk (faster)",
    )
    args = parser.parse_args()
    root = args.spdmx_root or Path(SPDMX_ROOT)
    if not args.spdmx_root and not root.is_dir():
        root = Path(OUTPUT_DIR) / SPDMX_DATASET_DIR_NAME
    if args.enrich_lengths_only:
        dest = enrich_songs_csv_song_lengths(root, jobs=args.jobs)
        songs = pd.read_csv(dest)
        n = int(songs[SONG_LENGTH_COLUMN].notna().sum()) if SONG_LENGTH_COLUMN in songs else 0
        print(f"wrote {dest} ({n}/{len(songs)} with song_length)")
        return
    dest = write_songs_table(
        root,
        out_path=args.out,
        check_files=not args.no_check_files,
    )
    summary_path = dest.with_suffix(".summary.json")
    print(summary_path.read_text())
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
