"""Tests for synthesis.progress pre-realify progress reporting."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from synthesis.pass_tables import pass_recipe_csv, pass_stems_csv
from synthesis.paths import MIDI_INDEX_FILE_NAME
from synthesis.progress import (
    MISSING_STEMS_REPORT_NAME,
    MissingClaimedStem,
    count_claimed_on_disk,
    format_remove_missing_commands,
    handle_missing_claimed_stems,
    parse_args,
    remove_missing_claimed_stems,
    report_pre_realify_progress,
    write_missing_stems_report,
)


def test_count_pass_progress_credits_polyphony_fallbacks(tmp_path: Path, capsys):
    from synthesis.progress import count_pass_progress_fast
    from synthesis.recipe import DEFAULT_RECIPE_PATH, load_recipe

    out = tmp_path / "out"
    tables = out / "dev" / "final"
    media = out / "SPDMX"
    tables.mkdir(parents=True)
    (media / "raw").mkdir(parents=True)
    song_a = str(media / "raw" / "1" / "11" / "QmA")
    song_b = str(media / "raw" / "2" / "22" / "QmB")
    pd.DataFrame({
        "song_id": ["1/11/QmA", "2/22/QmB"],
        "mid": ["x.mid", "y.mid"],
        "n_tracks": [2, 2],
        "n_fluidsynth": [1, 0],
        "n_ddsp_piano": [0, 0],
        "n_midi_ddsp": [1, 2],
    }).to_csv(tables / MIDI_INDEX_FILE_NAME, index=False)
    pd.DataFrame([{
        "path": song_a, "track": 0, "category": "piano", "ablation": "basic",
        "method": "basic", "fallback": "basic", "backend": "fluidsynth",
        "realify": False, "reason": None,
    }, {
        "path": song_b, "track": 1, "category": "strings", "ablation": "ddsp_basic",
        "method": "midi-ddsp", "fallback": "basic", "backend": "fluidsynth",
        "realify": False, "reason": "soundfont_polyphonic",
    }]).to_csv(pass_recipe_csv(tables, "fluidsynth"), index=False)
    pd.DataFrame([{
        "path": song_b, "track": 0, "category": "strings", "ablation": "ddsp_basic",
        "method": "midi-ddsp", "fallback": "basic", "backend": "midi_ddsp",
        "realify": False, "reason": "midi_ddsp_eligible",
    }]).to_csv(pass_recipe_csv(tables, "midi_ddsp"), index=False)

    rows = count_pass_progress_fast(tables, load_recipe(DEFAULT_RECIPE_PATH))
    by_pass = {r["pass"]: r for r in rows}
    md = by_pass["midi_ddsp"]
    assert md["neural_done"] == 1
    assert md["fallback_done"] == 1
    assert md["remaining"] == 1  # song A still needs its 1 midi_ddsp track
    assert md["assigned"] == 2  # renderable denominator, not layout 3

    report_pre_realify_progress(
        output_dir=str(out),
        recipe_path=DEFAULT_RECIPE_PATH,
        check_disk=False,
    )
    text = capsys.readouterr().out
    assert "soundfont_polyphonic" in text
    assert "Polyphony fallbacks" in text


def test_count_pass_progress_ignores_pending_midi_ddsp(tmp_path: Path):
    from synthesis.progress import count_pass_progress_fast
    from synthesis.recipe import DEFAULT_RECIPE_PATH, load_recipe

    out = tmp_path / "out"
    tables = out / "dev" / "final"
    media = out / "SPDMX"
    tables.mkdir(parents=True)
    (media / "raw").mkdir(parents=True)
    song = str(media / "raw" / "1" / "11" / "QmA")
    pd.DataFrame({
        "song_id": ["1/11/QmA"],
        "mid": ["x.mid"],
        "n_tracks": [2],
        "n_fluidsynth": [0],
        "n_ddsp_piano": [0],
        "n_midi_ddsp": [2],
    }).to_csv(tables / MIDI_INDEX_FILE_NAME, index=False)
    pd.DataFrame([{
        "path": song, "track": 0, "category": "strings", "ablation": "ddsp_basic",
        "method": "midi-ddsp", "fallback": "basic", "backend": "pending_midi_ddsp",
        "realify": False, "reason": "midi_ddsp_eligible",
    }, {
        "path": song, "track": 1, "category": "strings", "ablation": "ddsp_basic",
        "method": "midi-ddsp", "fallback": "basic", "backend": "fluidsynth",
        "realify": False, "reason": "soundfont_polyphonic",
    }]).to_csv(pass_recipe_csv(tables, "fluidsynth"), index=False)

    rows = count_pass_progress_fast(tables, load_recipe(DEFAULT_RECIPE_PATH))
    md = {r["pass"]: r for r in rows}["midi_ddsp"]
    assert md["fallback_done"] == 1
    assert md["remaining"] == 1
    assert md["neural_done"] == 0


def test_report_pre_realify_progress_csv_only(tmp_path: Path, capsys):
    out = tmp_path / "out"
    tables = out / "dev" / "final"
    media = out / "SPDMX"
    tables.mkdir(parents=True)
    (media / "raw").mkdir(parents=True)
    pd.DataFrame({
        "song_id": ["1/11/QmA", "2/22/QmB"],
        "mid": ["x.mid", "y.mid"],
        "n_tracks": [2, 1],
        "n_fluidsynth": [2, 0],
        "n_ddsp_piano": [0, 0],
        "n_midi_ddsp": [0, 1],
    }).to_csv(tables / MIDI_INDEX_FILE_NAME, index=False)
    path_a = str(media / "raw" / "1" / "11" / "QmA")
    pd.DataFrame([{
        "path": path_a,
        "track": 0,
        "category": "piano",
        "ablation": "basic",
        "method": "basic",
        "fallback": "basic",
        "backend": "fluidsynth",
        "realify": False,
    }]).to_csv(pass_recipe_csv(tables, "fluidsynth"), index=False)

    from synthesis.recipe import DEFAULT_RECIPE_PATH

    rows = report_pre_realify_progress(
        output_dir=str(out),
        recipe_path=DEFAULT_RECIPE_PATH,
        check_disk=False,
    )
    by_pass = {r["pass"]: r for r in rows}
    assert "fluidsynth" in by_pass
    assert by_pass["fluidsynth"]["assigned"] == 2
    assert by_pass["fluidsynth"]["recipe_done"] == 1
    assert by_pass["fluidsynth"]["remaining"] == 1
    out_text = capsys.readouterr().out
    assert "Pre-realify recipe progress" in out_text


def test_count_claimed_on_disk(tmp_path: Path):
    import numpy as np
    import soundfile as sf
    from shared.config import SAMPLE_RATE

    tables = tmp_path / "final"
    tables.mkdir()
    song = tmp_path / "raw" / "a" / "b" / "Qm"
    song.mkdir(parents=True)
    path = str(song)
    sf.write(str(song / "0.flac"), np.zeros(100, np.float32), SAMPLE_RATE, format="FLAC")
    pd.DataFrame({
        "path": [path, path],
        "track": [0, 1],
    }).to_csv(pass_stems_csv(tables, "fluidsynth"), index=False)
    on_disk, claimed = count_claimed_on_disk(tables, "fluidsynth", jobs=1)
    assert claimed == 2
    assert on_disk == 1
    on_disk2, claimed2 = count_claimed_on_disk(tables, "fluidsynth", jobs=2)
    assert (on_disk2, claimed2) == (1, 2)
    on_strict, _ = count_claimed_on_disk(tables, "fluidsynth", strict=True, jobs=2)
    assert on_strict == 1


def test_write_missing_stems_report(tmp_path: Path):
    tables = tmp_path / "final"
    tables.mkdir()
    song = str(tmp_path / "raw" / "1" / "11" / "QmA")
    missing = [
        MissingClaimedStem("midi_ddsp", song, 4, f"{song}/4.flac"),
    ]
    report = write_missing_stems_report(tables, missing)
    assert report.name == MISSING_STEMS_REPORT_NAME
    text = report.read_text(encoding="utf-8")
    assert "midi_ddsp" in text
    assert "1/11/QmA" in text
    assert "track" in text.splitlines()[0]


def test_format_remove_missing_commands(tmp_path: Path):
    tables = tmp_path / "final"
    tables.mkdir()
    song = str(tmp_path / "raw" / "1" / "11" / "QmA")
    pd.DataFrame({"path": [song], "track": [4]}).to_csv(
        pass_stems_csv(tables, "midi_ddsp"),
        index=False,
    )
    missing = [
        MissingClaimedStem("midi_ddsp", song, 4, f"{song}/4.flac"),
    ]
    commands = format_remove_missing_commands(tables, missing)
    assert f'rm -v "{song}/4.flac"' in commands
    assert "sed -i" in commands
    assert "stems.midi_ddsp.csv" in commands


def test_remove_missing_claimed_stems(tmp_path: Path):
    tables = tmp_path / "final"
    tables.mkdir()
    song_dir = tmp_path / "raw" / "1" / "11" / "QmA"
    song_dir.mkdir(parents=True)
    song = str(song_dir)
    bad_flac = song_dir / "4.flac"
    bad_flac.write_bytes(b"")
    pd.DataFrame({
        "path": [song],
        "track": [4],
        "category": "strings",
        "ablation": "basic",
        "method": "basic",
        "fallback": "basic",
        "backend": "midi_ddsp",
        "realify": False,
    }).to_csv(pass_recipe_csv(tables, "midi_ddsp"), index=False)
    pd.DataFrame({
        "path": [song],
        "track": [4],
        "original_track": [4],
        "program": [41],
        "is_drum": [False],
        "name": ["Viola"],
        "has_lyrics": [False],
        "max_velocity": [80],
        "velocity_scale": [1.0],
    }).to_csv(pass_stems_csv(tables, "midi_ddsp"), index=False)

    missing = [MissingClaimedStem("midi_ddsp", song, 4, str(bad_flac))]
    flacs, rows = remove_missing_claimed_stems(tables, missing)
    assert flacs == 1
    assert rows == 2
    assert not bad_flac.is_file()
    assert pass_stems_csv(tables, "midi_ddsp").read_text(encoding="utf-8").count("\n") == 1
    assert pass_recipe_csv(tables, "midi_ddsp").read_text(encoding="utf-8").count("\n") == 1


def test_handle_missing_claimed_stems_prints_commands(tmp_path: Path, capsys, monkeypatch):
    tables = tmp_path / "final"
    tables.mkdir()
    song = str(tmp_path / "raw" / "1" / "11" / "QmA")
    pd.DataFrame({"path": [song], "track": [4]}).to_csv(
        pass_stems_csv(tables, "midi_ddsp"),
        index=False,
    )
    missing = [MissingClaimedStem("midi_ddsp", song, 4, f"{song}/4.flac")]
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    handle_missing_claimed_stems(tables, missing, auto_yes=False)
    out = capsys.readouterr().out
    assert MISSING_STEMS_REPORT_NAME in out
    assert "Manual cleanup commands" in out
    assert "rm -v" in out
    assert "sed -i" in out


def test_parse_args_defaults():
    args = parse_args([])
    assert not args.no_check_disk
    assert not args.no_strict_disk
    assert not args.yes


def test_parse_args_no_check_disk():
    args = parse_args(["--no-check-disk"])
    assert args.no_check_disk

