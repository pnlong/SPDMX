"""Extract and aggregate MIDI track names from PDMX."""

from __future__ import annotations

from pathlib import Path

import mido
import pandas as pd

from analysis.pdmx_subset import filter_pdmx_subset
from shared.csv_tables import sanitize_track_name
from synthesis.patches import _gm_class, resolve_probe_category

UNNAMED_TRACK = "(unnamed)"


def normalize_track_name(name: str | None) -> str:
    cleaned = sanitize_track_name(name)
    if not cleaned:
        return UNNAMED_TRACK
    return cleaned.strip().lower()


def mid_path_for_row(mid_rel: str, pdmx_root: str | Path) -> Path:
    pdmx_root = Path(pdmx_root)
    if mid_rel.startswith("/"):
        return Path(str(pdmx_root) + mid_rel)
    return pdmx_root / mid_rel.lstrip("/")


def load_pdmx_mid_paths(
    dataset_filepath: str | Path,
    *,
    subset: str = "all_valid",
) -> tuple[pd.Series, Path]:
    """Return MIDI path column and PDMX root directory for a chosen subset."""
    dataset_filepath = Path(dataset_filepath)
    pdmx_root = dataset_filepath.parent

    usecols = ["mid", "subset:all_valid"]
    if subset == "rated_deduplicated":
        from shared.config import ABLATION_SUBSET_COLUMN

        usecols.append(ABLATION_SUBSET_COLUMN)

    dataset = pd.read_csv(dataset_filepath, usecols=usecols)
    filtered = filter_pdmx_subset(dataset, subset)

    return filtered["mid"].reset_index(drop=True), pdmx_root


def extract_named_stems_from_mid(mid_path: str | Path) -> list[dict] | None:
    """Parse a MIDI file and return one record per non-empty track."""
    try:
        midi = mido.MidiFile(filename=str(mid_path), charset="utf8")
    except Exception:
        return None

    rows: list[dict] = []
    for track in midi.tracks:
        program = 0
        is_drum = False
        track_name: str | None = None
        n_notes = 0
        determined_whether_track_is_drum = False

        for message in track:
            if message.type == "note_on" and message.velocity > 0:
                n_notes += 1
            elif message.type == "program_change":
                program = message.program
            elif message.type == "track_name":
                track_name = sanitize_track_name(message.name)
            if not determined_whether_track_is_drum and hasattr(message, "channel"):
                is_drum = message.channel == 9
                determined_whether_track_is_drum = True

        if n_notes == 0:
            continue

        normalized = normalize_track_name(track_name)
        rows.append({
            "track_name": normalized,
            "display_name": track_name or UNNAMED_TRACK,
            "program": int(program),
            "is_drum": bool(is_drum),
            "gm_class": _gm_class(program, is_drum),
            "category": resolve_probe_category(
                program=int(program),
                is_drum=bool(is_drum),
                track_name=track_name if track_name and len(track_name) > 0 else None,
            ),
        })

    return rows


def stems_dataframe(stem_records: list[dict]) -> pd.DataFrame:
    if not stem_records:
        return pd.DataFrame(
            columns=["track_name", "display_name", "program", "is_drum", "gm_class", "category"]
        )
    return pd.DataFrame(stem_records)


def build_track_name_report(
    stems: pd.DataFrame,
    *,
    subset: str,
    n_songs: int,
    n_songs_failed: int,
) -> dict:
    n_stems = len(stems)
    name_counts = stems["track_name"].value_counts()
    track_name_rows = []
    for track_name, count in name_counts.items():
        track_name_rows.append({
            "track_name": str(track_name),
            "count": int(count),
            "pct": round(100 * count / n_stems, 4) if n_stems else 0.0,
        })

    category_counts = {
        key: int(value)
        for key, value in stems["category"].value_counts().items()
    }
    gm_class_counts = {
        key: int(value)
        for key, value in stems["gm_class"].value_counts().items()
    }

    unnamed = int((stems["track_name"] == UNNAMED_TRACK).sum()) if n_stems else 0
    named = n_stems - unnamed

    return {
        "subset": subset,
        "source": "MIDI track_name meta-events (non-empty tracks)",
        "n_songs": n_songs,
        "n_songs_failed": n_songs_failed,
        "n_stems": n_stems,
        "avg_stems_per_song": round(n_stems / n_songs, 4) if n_songs else 0.0,
        "n_named_stems": named,
        "n_unnamed_stems": unnamed,
        "pct_named_stems": round(100 * named / n_stems, 4) if n_stems else 0.0,
        "pct_unnamed_stems": round(100 * unnamed / n_stems, 4) if n_stems else 0.0,
        "n_unique_track_names": int(name_counts.shape[0]),
        "track_names": track_name_rows,
        "listening_categories": category_counts,
        "gm_classes": gm_class_counts,
    }


def print_track_name_report(report: dict, *, top_n: int = 50) -> None:
    print(f"Subset: {report['subset']}")
    print(f"Source: {report['source']}")
    print(
        f"Songs: {report['n_songs']:,} "
        f"(missing/unreadable MIDI: {report['n_songs_failed']:,})"
    )
    print(
        f"Tracks (non-empty MIDI tracks): {report['n_stems']:,} "
        f"({report['avg_stems_per_song']:.2f} per song)"
    )
    print(
        f"Named: {report['n_named_stems']:,} ({report['pct_named_stems']:.1f}%)  "
        f"Unnamed: {report['n_unnamed_stems']:,} ({report['pct_unnamed_stems']:.1f}%)"
    )
    print(f"Unique track names: {report['n_unique_track_names']}")

    print(f"\n--- Track names (top {top_n}, sorted by count) ---")
    names = sorted(report["track_names"], key=lambda row: (-row["count"], row["track_name"]))
    for row in names[:top_n]:
        print(f"  {row['track_name']:32s} {row['count']:7,d}  ({row['pct']:5.2f}%)")

    print("\n--- Listening categories (preset routing from names + GM program) ---")
    for name, count in sorted(
        report["listening_categories"].items(),
        key=lambda item: (-item[1], item[0]),
    ):
        pct = 100 * count / report["n_stems"]
        print(f"  {name:14s} {count:7,d}  ({pct:5.1f}%)")

    print("\n--- GM instrument classes ---")
    for name, count in sorted(
        report["gm_classes"].items(),
        key=lambda item: (-item[1], item[0]),
    ):
        pct = 100 * count / report["n_stems"]
        print(f"  {name:22s} {count:7,d}  ({pct:5.1f}%)")
