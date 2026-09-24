"""Extract and aggregate General MIDI program usage from PDMX MIDI files."""

from __future__ import annotations

from pathlib import Path

import mido
import pandas as pd

from analysis.pdmx_subset import filter_pdmx_subset
from shared.csv_tables import sanitize_track_name
from synthesis.patches import _gm_class, resolve_probe_category

TRACKS_DELIMITER = "-"

# Sentinel gm_id for channel-10 drum kits (not a melodic GM program).
DRUM_GM_ID = 128

# Standard General MIDI melodic program names (0–127).
GM_PROGRAM_NAMES: tuple[str, ...] = (
    "Acoustic Grand Piano",
    "Bright Acoustic Piano",
    "Electric Grand Piano",
    "Honky-tonk Piano",
    "Electric Piano 1",
    "Electric Piano 2",
    "Harpsichord",
    "Clavinet",
    "Celesta",
    "Glockenspiel",
    "Music Box",
    "Vibraphone",
    "Marimba",
    "Xylophone",
    "Tubular Bells",
    "Dulcimer",
    "Drawbar Organ",
    "Percussive Organ",
    "Rock Organ",
    "Church Organ",
    "Reed Organ",
    "Accordion",
    "Harmonica",
    "Tango Accordion",
    "Acoustic Guitar (nylon)",
    "Acoustic Guitar (steel)",
    "Electric Guitar (jazz)",
    "Electric Guitar (clean)",
    "Electric Guitar (muted)",
    "Overdriven Guitar",
    "Distortion Guitar",
    "Guitar Harmonics",
    "Acoustic Bass",
    "Electric Bass (finger)",
    "Electric Bass (pick)",
    "Fretless Bass",
    "Slap Bass 1",
    "Slap Bass 2",
    "Synth Bass 1",
    "Synth Bass 2",
    "Violin",
    "Viola",
    "Cello",
    "Contrabass",
    "Tremolo Strings",
    "Pizzicato Strings",
    "Orchestral Harp",
    "Timpani",
    "String Ensemble 1",
    "String Ensemble 2",
    "Synth Strings 1",
    "Synth Strings 2",
    "Choir Aahs",
    "Voice Oohs",
    "Synth Voice",
    "Orchestra Hit",
    "Trumpet",
    "Trombone",
    "Tuba",
    "Muted Trumpet",
    "French Horn",
    "Brass Section",
    "Synth Brass 1",
    "Synth Brass 2",
    "Soprano Sax",
    "Alto Sax",
    "Tenor Sax",
    "Baritone Sax",
    "Oboe",
    "English Horn",
    "Bassoon",
    "Clarinet",
    "Piccolo",
    "Flute",
    "Recorder",
    "Pan Flute",
    "Blown Bottle",
    "Shakuhachi",
    "Whistle",
    "Ocarina",
    "Lead 1 (square)",
    "Lead 2 (sawtooth)",
    "Lead 3 (calliope)",
    "Lead 4 (chiff)",
    "Lead 5 (charang)",
    "Lead 6 (voice)",
    "Lead 7 (fifths)",
    "Lead 8 (bass + lead)",
    "Pad 1 (new age)",
    "Pad 2 (warm)",
    "Pad 3 (polysynth)",
    "Pad 4 (choir)",
    "Pad 5 (bowed)",
    "Pad 6 (metallic)",
    "Pad 7 (halo)",
    "Pad 8 (sweep)",
    "FX 1 (rain)",
    "FX 2 (soundtrack)",
    "FX 3 (crystal)",
    "FX 4 (atmosphere)",
    "FX 5 (brightness)",
    "FX 6 (goblins)",
    "FX 7 (echoes)",
    "FX 8 (sci-fi)",
    "Sitar",
    "Banjo",
    "Shamisen",
    "Koto",
    "Kalimba",
    "Bagpipe",
    "Fiddle",
    "Shanai",
    "Tinkle Bell",
    "Agogo",
    "Steel Drums",
    "Woodblock",
    "Taiko Drum",
    "Melodic Tom",
    "Synth Drum",
    "Reverse Cymbal",
    "Guitar Fret Noise",
    "Breath Noise",
    "Seashore",
    "Bird Tweet",
    "Telephone Ring",
    "Helicopter",
    "Applause",
    "Gunshot",
)


def gm_program_name(program: int) -> str:
    if 0 <= program < len(GM_PROGRAM_NAMES):
        return GM_PROGRAM_NAMES[program]
    return f"Unknown program {program}"


def gm_id_label(gm_id_value: int) -> str:
    if int(gm_id_value) == DRUM_GM_ID:
        return f"{DRUM_GM_ID}: Drums (channel 10)"
    if 0 <= gm_id_value < len(GM_PROGRAM_NAMES):
        return f"{gm_id_value}: {GM_PROGRAM_NAMES[gm_id_value]}"
    return f"{gm_id_value}: Unknown"


def gm_program_display_name(gm_id_value: int) -> str:
    """Short instrument name for plots (drums omit the channel-10 note)."""
    if int(gm_id_value) == DRUM_GM_ID:
        return "Drums"
    return gm_program_name(int(gm_id_value))


def gm_program_paper_label(gm_id_value: int) -> str:
    """Paper-style label: ``{instrument name} ({program})``; drums are just ``Drums``."""
    gm_id = int(gm_id_value)
    if gm_id == DRUM_GM_ID:
        return "Drums"
    return f"{gm_program_display_name(gm_id)} ({gm_id})"


def parse_tracks_cell(tracks) -> list[int] | None:
    """Split a PDMX ``tracks`` cell on ``-`` and return GM program ids."""
    if pd.isna(tracks):
        return None
    text = str(tracks).strip()
    if not text:
        return None
    try:
        return [int(part) for part in text.split(TRACKS_DELIMITER) if part.strip()]
    except ValueError:
        return None


def program_to_stem_record(program: int, presets: dict | None = None, *, is_drum: bool = False) -> dict:
    del presets  # unused; kept for call-site compatibility
    gm_id = DRUM_GM_ID if is_drum else int(program)
    return {
        "gm_id": gm_id,
        "program": int(program),
        "is_drum": bool(is_drum),
        "gm_class": _gm_class(program, is_drum),
        "category": resolve_probe_category(
            program=int(program), is_drum=bool(is_drum), track_name=None,
        ),
    }


def tracks_cell_to_stem_records(tracks, presets: dict | None = None) -> list[dict]:
    """Legacy helper: melodic programs only (no channel-10 drum detection)."""
    del presets
    programs = parse_tracks_cell(tracks)
    if programs is None:
        return []
    return [program_to_stem_record(program, is_drum=False) for program in programs]


def extract_gm_stems_from_mid(mid_path: str | Path, presets: dict | None = None) -> list[dict] | None:
    """Parse a MIDI file; one record per non-empty track (drums = channel 10)."""
    del presets
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

        record = program_to_stem_record(program, is_drum=is_drum)
        record["name"] = track_name if track_name else None
        record["category"] = resolve_probe_category(
            program=int(program),
            is_drum=bool(is_drum),
            track_name=track_name if track_name else None,
        )
        rows.append(record)

    return rows


def load_pdmx_tracks(
    dataset_filepath: str | Path,
    *,
    subset: str = "all_valid",
) -> pd.Series:
    """Return the ``tracks`` column for a chosen PDMX subset."""
    dataset_filepath = Path(dataset_filepath)

    usecols = ["tracks", "subset:all_valid"]
    if subset == "rated_deduplicated":
        from shared.config import ABLATION_SUBSET_COLUMN

        usecols.append(ABLATION_SUBSET_COLUMN)

    dataset = pd.read_csv(dataset_filepath, usecols=usecols)
    filtered = filter_pdmx_subset(dataset, subset)

    return filtered["tracks"].reset_index(drop=True)


def stems_dataframe(stem_records: list[dict]) -> pd.DataFrame:
    columns = ["gm_id", "program", "is_drum", "gm_class", "category", "name"]
    if not stem_records:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(stem_records)


def stems_from_register(
    register: pd.DataFrame,
    presets: dict | None = None,
) -> pd.DataFrame:
    """Build GM stem records from a register table using ``program_corrected``."""
    del presets
    if register is None or len(register) == 0:
        return stems_dataframe([])

    records: list[dict] = []
    for _, row in register.iterrows():
        is_drum = bool(row.get("is_drum", False))
        program = int(row["program_corrected"])
        name = row.get("name")
        name = None if pd.isna(name) else name
        gm_id = DRUM_GM_ID if is_drum else int(program)
        records.append(
            {
                "gm_id": gm_id,
                "program": int(program),
                "is_drum": bool(is_drum),
                "gm_class": _gm_class(program, is_drum),
                "category": resolve_probe_category(
                    program=int(program), is_drum=bool(is_drum), track_name=name,
                ),
                "name": name,
            }
        )
    return stems_dataframe(records)


def build_gm_report(
    stems: pd.DataFrame,
    *,
    subset: str,
    n_songs: int,
    n_songs_failed: int,
    source: str = "MIDI files (program_change + channel-10 drums)",
) -> dict:
    n_stems = len(stems)
    gm_counts = stems["gm_id"].value_counts().sort_index()
    program_rows = []
    for gm_id_value, count in gm_counts.items():
        gm_id_int = int(gm_id_value)
        if gm_id_int == DRUM_GM_ID:
            name = "Drums (channel 10)"
        else:
            name = gm_program_name(gm_id_int)
        program_rows.append({
            "gm_id": gm_id_int,
            "program": gm_id_int if gm_id_int != DRUM_GM_ID else None,
            "name": name,
            "label": gm_id_label(gm_id_int),
            "count": int(count),
            "pct": round(100 * count / n_stems, 4) if n_stems else 0.0,
        })

    gm_class_counts = {
        key: int(value)
        for key, value in stems["gm_class"].value_counts().items()
    }
    category_counts = {
        key: int(value)
        for key, value in stems["category"].value_counts().items()
    }

    if n_stems and "is_drum" in stems.columns:
        n_drums = int(stems["is_drum"].sum())
        melodic = stems[~stems["is_drum"].astype(bool)]
        program_0_count = int((melodic["program"] == 0).sum())
    else:
        n_drums = int((stems["gm_id"] == DRUM_GM_ID).sum()) if n_stems else 0
        program_0_count = int((stems["program"] == 0).sum()) if n_stems else 0

    return {
        "subset": subset,
        "source": source,
        "n_songs": n_songs,
        "n_songs_failed": n_songs_failed,
        "n_stems": n_stems,
        "avg_stems_per_song": round(n_stems / n_songs, 4) if n_songs else 0.0,
        "n_unique_gm_ids": int(gm_counts.shape[0]),
        "n_drum_stems": n_drums,
        "drum_pct_of_stems": round(100 * n_drums / n_stems, 4) if n_stems else 0.0,
        "program_0_count": program_0_count,
        "program_0_pct_of_stems": round(100 * program_0_count / n_stems, 4) if n_stems else 0.0,
        "gm_programs": program_rows,
        "gm_classes": gm_class_counts,
        "listening_categories": category_counts,
    }


def print_gm_report(report: dict) -> None:
    print(f"Subset: {report['subset']}")
    print(f"Source: {report['source']}")
    print(f"Songs: {report['n_songs']:,} (missing/unreadable MIDI: {report['n_songs_failed']:,})")
    print(
        f"Tracks (non-empty MIDI tracks): {report['n_stems']:,} "
        f"({report['avg_stems_per_song']:.2f} per song)"
    )
    print(f"Unique GM ids present (incl. drums sentinel {DRUM_GM_ID}): {report['n_unique_gm_ids']}")
    print(
        f"Drums (channel 10): {report['n_drum_stems']:,} "
        f"({report['drum_pct_of_stems']:.1f}% of all tracks)"
    )
    print(
        f"Program 0 (Acoustic Grand Piano, non-drum): {report['program_0_count']:,} "
        f"({report['program_0_pct_of_stems']:.1f}% of all tracks)"
    )

    print("\n--- GM program ids (sorted by count; drums = channel 10) ---")
    programs = sorted(report["gm_programs"], key=lambda row: (-row["count"], row["gm_id"]))
    for row in programs:
        print(f"  {row['label']:42s} {row['count']:7,d}  ({row['pct']:5.2f}%)")

    print("\n--- GM instrument classes ---")
    for name, count in sorted(
        report["gm_classes"].items(),
        key=lambda item: (-item[1], item[0]),
    ):
        pct = 100 * count / report["n_stems"]
        print(f"  {name:22s} {count:7,d}  ({pct:5.1f}%)")

    if report["listening_categories"]:
        print("\n--- Listening categories (preset routing) ---")
        for name, count in sorted(
            report["listening_categories"].items(),
            key=lambda item: (-item[1], item[0]),
        ):
            pct = 100 * count / report["n_stems"]
            print(f"  {name:14s} {count:7,d}  ({pct:5.1f}%)")
