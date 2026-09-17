"""Aggregate song-hours by stem count for SPDMX and Slakh2100.

Song-hours = per-song mix duration once (never multiplied by stem count).
Multitrack = n_stems >= 2.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
import soundfile as sf
import yaml

from shared.config import OUTPUT_DIR, SLAKH_ROOT, SPDMX_DATASET_DIR_NAME

REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER_DATA = REPO_ROOT / "analysis" / "paper_data"
DOCS_DATA = REPO_ROOT / "docs" / "data"
DEFAULT_SPDMX_SONGS = Path(OUTPUT_DIR) / SPDMX_DATASET_DIR_NAME / "songs.csv"
DEFAULT_SLAKH = Path(SLAKH_ROOT)
MAX_BIN = 20


def _bin_hours(hours_by_n: dict[int, float], *, max_bin: int = MAX_BIN) -> dict:
    labels: list[str] = []
    hours: list[float] = []
    for n in range(2, max_bin):
        labels.append(str(n))
        hours.append(round(float(hours_by_n.get(n, 0.0)), 4))
    overflow = sum(v for k, v in hours_by_n.items() if k >= max_bin)
    labels.append(f"{max_bin}+")
    hours.append(round(float(overflow), 4))
    return {"labels": labels, "hours": hours}


def spdmx_hours_by_stems(
    songs_csv: Path,
    *,
    max_bin: int = MAX_BIN,
) -> dict:
    songs = pd.read_csv(songs_csv, usecols=["n_tracks", "song_length"])
    songs["n_tracks"] = pd.to_numeric(songs["n_tracks"], errors="coerce")
    songs["song_length"] = pd.to_numeric(songs["song_length"], errors="coerce")
    songs = songs.dropna(subset=["n_tracks", "song_length"])
    songs = songs[songs["song_length"] > 0]
    songs["n_tracks"] = songs["n_tracks"].astype(int)

    total_songs = int(len(songs))
    total_hours = float(songs["song_length"].sum() / 3600.0)
    multi = songs[songs["n_tracks"] >= 2]
    multi_songs = int(len(multi))
    multi_hours = float(multi["song_length"].sum() / 3600.0)

    hours_by_n: dict[int, float] = defaultdict(float)
    counts_by_n: dict[int, int] = defaultdict(int)
    for n, length in zip(multi["n_tracks"], multi["song_length"]):
        hours_by_n[int(n)] += float(length) / 3600.0
        counts_by_n[int(n)] += 1

    binned = _bin_hours(hours_by_n, max_bin=max_bin)
    return {
        "corpus": "SPDMX",
        "total_songs": total_songs,
        "total_hours": round(total_hours, 1),
        "multitrack_songs": multi_songs,
        "multitrack_hours": round(multi_hours, 1),
        "multitrack_song_frac": round(multi_songs / total_songs, 4) if total_songs else 0.0,
        "multitrack_hour_frac": round(multi_hours / total_hours, 4) if total_hours else 0.0,
        "labels": binned["labels"],
        "hours": binned["hours"],
        "counts_by_n": {str(k): int(v) for k, v in sorted(counts_by_n.items())},
        "note": "Song-hours = mix duration per song; single-stem songs omitted from bars.",
    }


def _slakh_stem_count(meta: dict) -> int:
    stems = meta.get("stems") or {}
    n = 0
    for s in stems.values():
        if isinstance(s, dict) and s.get("audio_rendered") is False:
            continue
        n += 1
    return n if n else len(stems)


def slakh_hours_by_stems(
    slakh_root: Path,
    *,
    include_omitted: bool = True,
    max_bin: int = MAX_BIN,
    cache_path: Path | None = None,
) -> dict:
    """Scan Slakh track dirs; optionally cache per-track rows."""
    if cache_path and cache_path.is_file():
        rows = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        splits = ["train", "validation", "test"]
        if include_omitted:
            splits.append("omitted")
        rows = []
        for split in splits:
            split_dir = slakh_root / split
            if not split_dir.is_dir():
                continue
            for track in sorted(split_dir.iterdir()):
                if not track.is_dir():
                    continue
                meta_path = track / "metadata.yaml"
                mix = track / "mix.flac"
                if not meta_path.is_file() or not mix.is_file():
                    continue
                meta = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
                n = _slakh_stem_count(meta or {})
                dur = float(sf.info(str(mix)).duration)
                rows.append(
                    {
                        "track": track.name,
                        "split": split,
                        "n_stems": int(n),
                        "duration_sec": dur,
                    }
                )
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")

    hours_by_n: dict[int, float] = defaultdict(float)
    counts_by_n: dict[int, int] = defaultdict(int)
    total_hours = 0.0
    for row in rows:
        n = int(row["n_stems"])
        h = float(row["duration_sec"]) / 3600.0
        hours_by_n[n] += h
        counts_by_n[n] += 1
        total_hours += h

    total_songs = len(rows)
    # Slakh mixes are always multi-stem in practice; still filter >= 2 for consistency.
    multi_hours = sum(v for k, v in hours_by_n.items() if k >= 2)
    multi_songs = sum(c for k, c in counts_by_n.items() if k >= 2)
    binned = _bin_hours(hours_by_n, max_bin=max_bin)
    return {
        "corpus": "Slakh2100",
        "include_omitted": include_omitted,
        "total_songs": total_songs,
        "total_hours": round(total_hours, 1),
        "multitrack_songs": int(multi_songs),
        "multitrack_hours": round(multi_hours, 1),
        "labels": binned["labels"],
        "hours": binned["hours"],
        "counts_by_n": {str(k): int(v) for k, v in sorted(counts_by_n.items())},
        "note": "Song-hours from mix.flac; all 2100 tracks when include_omitted=True.",
    }


def build_payload(
    *,
    songs_csv: Path,
    slakh_root: Path,
    include_omitted: bool = True,
    cache_path: Path | None = None,
    max_bin: int = MAX_BIN,
) -> dict:
    spdmx = spdmx_hours_by_stems(songs_csv, max_bin=max_bin)
    slakh = slakh_hours_by_stems(
        slakh_root,
        include_omitted=include_omitted,
        max_bin=max_bin,
        cache_path=cache_path,
    )
    # Align labels (SPDMX defines the shared axis).
    labels = spdmx["labels"]
    return {
        "labels": labels,
        "spdmx": spdmx,
        "slakh": slakh,
        "max_bin": max_bin,
        "y_axis": "song_hours",
        "definition": (
            "For each x=n, y is the sum of mix durations (hours) over songs with "
            "exactly n stems (overflow bin is n>=max_bin). Never multiply by stem count."
        ),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--songs-csv", type=Path, default=DEFAULT_SPDMX_SONGS)
    parser.add_argument("--slakh-root", type=Path, default=DEFAULT_SLAKH)
    parser.add_argument(
        "--no-omitted",
        action="store_true",
        help="Exclude Slakh omitted/ (redux only)",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=PAPER_DATA / "slakh_track_hours.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PAPER_DATA / "song_hours_by_stems.json",
    )
    parser.add_argument("--also-docs", action="store_true", default=True)
    args = parser.parse_args(argv)

    if not args.songs_csv.is_file():
        raise SystemExit(f"missing songs.csv: {args.songs_csv}")
    if not args.slakh_root.is_dir():
        raise SystemExit(f"missing Slakh root: {args.slakh_root}")

    payload = build_payload(
        songs_csv=args.songs_csv,
        slakh_root=args.slakh_root,
        include_omitted=not args.no_omitted,
        cache_path=args.cache,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    print(
        f"SPDMX total {payload['spdmx']['total_songs']} songs / "
        f"{payload['spdmx']['total_hours']} h; multitrack "
        f"{payload['spdmx']['multitrack_songs']} / {payload['spdmx']['multitrack_hours']} h"
    )
    print(
        f"Slakh {payload['slakh']['total_songs']} songs / {payload['slakh']['total_hours']} h"
    )
    if args.also_docs:
        docs_out = DOCS_DATA / "song_hours_by_stems.json"
        docs_out.parent.mkdir(parents=True, exist_ok=True)
        docs_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {docs_out}")


if __name__ == "__main__":
    main()
