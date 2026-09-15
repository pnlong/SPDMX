"""Tests for GM program analysis."""

from pathlib import Path

import matplotlib
import mido
import pandas as pd

matplotlib.use("Agg")

from analysis.analyze_gm_programs import analyze_gm_programs
from analysis.gm_programs import (
    DRUM_GM_ID,
    build_gm_report,
    extract_gm_stems_from_mid,
    gm_id_label,
    gm_program_name,
    parse_tracks_cell,
    tracks_cell_to_stem_records,
)
from analysis.plots import plot_gm_program_bar


def _write_minimal_midi(
    path: Path,
    *,
    program: int = 0,
    channel: int = 0,
    name: str | None = "Piano",
):
    midi = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    if name is not None:
        track.append(mido.MetaMessage("track_name", name=name, time=0))
    track.append(mido.Message("program_change", program=program, channel=channel, time=0))
    track.append(mido.Message("note_on", note=60, velocity=80, channel=channel, time=0))
    track.append(mido.Message("note_off", note=60, velocity=0, channel=channel, time=480))
    midi.tracks.append(track)
    midi.save(path)


def _write_piano_and_drums(path: Path):
    midi = mido.MidiFile(ticks_per_beat=480)

    piano = mido.MidiTrack()
    piano.append(mido.Message("program_change", program=0, channel=0, time=0))
    piano.append(mido.Message("note_on", note=60, velocity=80, channel=0, time=0))
    piano.append(mido.Message("note_off", note=60, velocity=0, channel=0, time=480))
    midi.tracks.append(piano)

    drums = mido.MidiTrack()
    drums.append(mido.Message("program_change", program=0, channel=9, time=0))
    drums.append(mido.Message("note_on", note=36, velocity=100, channel=9, time=0))
    drums.append(mido.Message("note_off", note=36, velocity=0, channel=9, time=480))
    midi.tracks.append(drums)

    midi.save(path)


def test_gm_id_labels():
    from analysis.gm_programs import gm_program_paper_label

    assert gm_program_name(0) == "Acoustic Grand Piano"
    assert gm_program_name(56) == "Trumpet"
    assert gm_id_label(0).startswith("0:")
    assert gm_id_label(56) == "56: Trumpet"
    assert "Drums" in gm_id_label(DRUM_GM_ID)
    assert gm_program_paper_label(0) == "Acoustic Grand Piano (0)"
    assert gm_program_paper_label(DRUM_GM_ID) == "Drums"


def test_parse_tracks_cell():
    assert parse_tracks_cell("0") == [0]
    assert parse_tracks_cell("0-42-43-48-48") == [0, 42, 43, 48, 48]
    assert parse_tracks_cell("") is None
    assert parse_tracks_cell(float("nan")) is None
    assert parse_tracks_cell("0-bad") is None


def test_tracks_cell_to_stem_records():
    rows = tracks_cell_to_stem_records("0-56")
    assert len(rows) == 2
    assert rows[0]["gm_id"] == 0
    assert rows[0]["gm_class"] == "piano"
    assert rows[1]["gm_id"] == 56
    assert rows[1]["gm_class"] == "brass"


def test_extract_gm_stems_includes_drums(tmp_path: Path):
    mid_path = tmp_path / "song.mid"
    _write_piano_and_drums(mid_path)
    rows = extract_gm_stems_from_mid(mid_path)
    assert rows is not None
    assert len(rows) == 2
    by_class = {row["gm_class"]: row for row in rows}
    assert by_class["piano"]["gm_id"] == 0
    assert by_class["drums"]["gm_id"] == DRUM_GM_ID
    assert by_class["drums"]["is_drum"] is True


def test_build_gm_report():
    stems = pd.DataFrame([
        {"gm_id": 0, "program": 0, "is_drum": False, "gm_class": "piano", "category": "default"},
        {"gm_id": DRUM_GM_ID, "program": 0, "is_drum": True, "gm_class": "drums", "category": "drums"},
        {"gm_id": 56, "program": 56, "is_drum": False, "gm_class": "brass", "category": "default"},
    ])
    report = build_gm_report(stems, subset="all_valid", n_songs=1, n_songs_failed=0)
    assert report["n_stems"] == 3
    assert report["n_drum_stems"] == 1
    assert report["program_0_count"] == 1  # excludes drum track with program 0
    assert any(row["gm_id"] == DRUM_GM_ID for row in report["gm_programs"])
    assert report["gm_classes"]["drums"] == 1


def test_stems_from_register_uses_program_corrected():
    from analysis.analyze_gm_programs import analyze_gm_programs_from_register
    from analysis.gm_programs import stems_from_register
    from analysis.gm_register import STATUS_CORRECTED, STATUS_KEEP, STATUS_SKIPPED_UNNAMED

    register = pd.DataFrame(
        [
            {
                "mid": "./a.mid",
                "track": 0,
                "name": "Harpsichord",
                "is_drum": False,
                "program_original": 0,
                "program_corrected": 6,
                "status": STATUS_CORRECTED,
            },
            {
                "mid": "./a.mid",
                "track": 1,
                "name": None,
                "is_drum": False,
                "program_original": 0,
                "program_corrected": 0,
                "status": STATUS_SKIPPED_UNNAMED,
            },
            {
                "mid": "./a.mid",
                "track": 2,
                "name": "Piano",
                "is_drum": False,
                "program_original": 0,
                "program_corrected": 0,
                "status": STATUS_KEEP,
            },
        ]
    )
    stems = stems_from_register(register)
    assert list(stems["program"]) == [6, 0, 0]

    report, filtered_stems = analyze_gm_programs_from_register(register, subset="all_valid")
    assert report["source"].startswith("GM register")
    assert len(filtered_stems) == 2  # unnamed excluded
    assert set(filtered_stems["program"]) == {6, 0}


def test_plot_gm_program_bar(tmp_path: Path):
    stems = pd.DataFrame([
        {"gm_id": 0, "program": 0, "is_drum": False, "gm_class": "piano", "category": "default"},
        {"gm_id": DRUM_GM_ID, "program": 0, "is_drum": True, "gm_class": "drums", "category": "drums"},
        {"gm_id": 56, "program": 56, "is_drum": False, "gm_class": "brass", "category": "default"},
    ])
    out = tmp_path / "gm_program_counts.png"
    plot_gm_program_bar(stems, out, top_n=10)
    assert out.exists()


def test_plot_gm_program_compare(tmp_path: Path):
    from analysis.plots import plot_gm_program_compare

    original = pd.DataFrame([
        {"gm_id": 0, "program": 0, "is_drum": False, "gm_class": "piano", "category": "default"},
        {"gm_id": 0, "program": 0, "is_drum": False, "gm_class": "piano", "category": "default"},
        {"gm_id": 56, "program": 56, "is_drum": False, "gm_class": "brass", "category": "default"},
        {"gm_id": DRUM_GM_ID, "program": 0, "is_drum": True, "gm_class": "drums", "category": "drums"},
    ])
    corrected = pd.DataFrame([
        {"gm_id": 6, "program": 6, "is_drum": False, "gm_class": "piano", "category": "default"},
        {"gm_id": 52, "program": 52, "is_drum": False, "gm_class": "ensemble", "category": "voice"},
        {"gm_id": 56, "program": 56, "is_drum": False, "gm_class": "brass", "category": "default"},
        {"gm_id": DRUM_GM_ID, "program": 0, "is_drum": True, "gm_class": "drums", "category": "drums"},
    ])
    out = tmp_path / "gm_program_counts_compare.png"
    plot_gm_program_compare(original, corrected, out, top_n=10)
    assert out.exists()

    split = tmp_path / "gm_program_counts_compare_by_source.png"
    plot_gm_program_compare(
        original,
        corrected,
        split,
        top_n=10,
        layout="by_source",
    )
    assert split.exists()


def test_plot_gm_stems_vs_hours(tmp_path: Path):
    from analysis.plots import plot_gm_stems_vs_hours

    summary = pd.DataFrame(
        [
            {
                "gm_id": 0,
                "label": "Acoustic Grand Piano (0)",
                "n_stems": 100,
                "wall_hours": 10.0,
                "active_hours": 8.0,
            },
            {
                "gm_id": 52,
                "label": "Choir Aahs (52)",
                "n_stems": 40,
                "wall_hours": 20.0,
                "active_hours": 12.0,
            },
            {"gm_id": DRUM_GM_ID, "label": "Drums", "n_stems": 20, "wall_hours": 5.0, "active_hours": 1.5},
        ]
    )
    out = tmp_path / "gm_stems_vs_hours.pdf"
    plot_gm_stems_vs_hours(summary, out, top_n=3, hours_col="active_hours")
    assert out.exists()


def test_plot_stems_per_song(tmp_path: Path):
    from analysis.plots import plot_stems_per_song

    out = tmp_path / "stems_per_song.pdf"
    plot_stems_per_song(
        {
            "labels": ["2", "3", "4", "20+"],
            "counts": [100, 40, 20, 5],
            "n_single": 200,
            "n_multi": 165,
            "n_total": 365,
        },
        out,
    )
    assert out.exists()


def test_plot_keeps_drums_outside_top_n(tmp_path: Path):
    rows = [
        {"gm_id": i, "program": i, "is_drum": False, "gm_class": "piano", "category": "default"}
        for i in range(5)
    ]
    rows.append({
        "gm_id": DRUM_GM_ID,
        "program": 0,
        "is_drum": True,
        "gm_class": "drums",
        "category": "drums",
    })
    # Make melodic ids dominate counts so drums fall outside top_n=3.
    stems = pd.DataFrame(rows + [
        {"gm_id": 0, "program": 0, "is_drum": False, "gm_class": "piano", "category": "default"},
        {"gm_id": 0, "program": 0, "is_drum": False, "gm_class": "piano", "category": "default"},
        {"gm_id": 1, "program": 1, "is_drum": False, "gm_class": "piano", "category": "default"},
    ])
    out = tmp_path / "gm_program_counts.png"
    plot_gm_program_bar(stems, out, top_n=3)
    assert out.exists()


def test_analyze_gm_programs_on_fake_pdmx(tmp_path: Path):
    pdmx_root = tmp_path / "PDMX"
    pdmx_root.mkdir()
    mid_rel = "/0/test/song.mid"
    mid_path = pdmx_root / "0/test/song.mid"
    mid_path.parent.mkdir(parents=True)
    _write_piano_and_drums(mid_path)

    csv_path = pdmx_root / "PDMX.csv"
    pd.DataFrame({
        "mid": [mid_rel, "/0/test/missing.mid"],
        "subset:all_valid": [True, True],
    }).to_csv(csv_path, index=False)

    report, stems = analyze_gm_programs(csv_path, subset="all_valid", jobs=1)
    assert report["n_songs"] == 2
    assert report["n_songs_failed"] == 1
    assert report["n_stems"] == 2
    assert report["n_drum_stems"] == 1
    assert (stems["gm_id"] == 0).sum() == 1
    assert (stems["gm_id"] == DRUM_GM_ID).sum() == 1
