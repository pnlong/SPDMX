"""Tests for duration analysis."""

from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from analysis.analyze_durations import analyze_stem_row, stem_duration_seconds
from analysis.report import build_report, duration_summary


def _write_flac(path: Path, seconds: float, sr: int = 44100):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.zeros((int(sr * seconds), 2)), sr, format="FLAC")


def test_stem_duration_seconds(tmp_path: Path):
    flac = tmp_path / "stem_0.flac"
    _write_flac(flac, 2.0)
    assert abs(stem_duration_seconds(flac) - 2.0) < 0.01


def test_analyze_stem_row(tmp_path: Path):
    song_dir = tmp_path / "song"
    _write_flac(song_dir / "stem_0.flac", 1.5)
    row = pd.Series({"path": str(song_dir), "track": 0, "program": 0, "is_drum": False, "name": "Piano"})
    result = analyze_stem_row(row)
    assert abs(result["duration_seconds"] - 1.5) < 0.01


def test_duration_summary():
    durations = pd.Series([30, 60, 90, 100, 110])
    summary = duration_summary(durations)
    assert summary["n_songs"] == 5
    assert summary["median_song_duration_seconds"] == 90.0


def test_build_report(tmp_path: Path):
    song_a = tmp_path / "a"
    song_b = tmp_path / "b"
    _write_flac(song_a / "stem_0.flac", 1.0)
    _write_flac(song_b / "stem_0.flac", 3.0)
    df = pd.DataFrame([
        {"path": str(song_a), "track": 0, "duration_seconds": 1.0, "program": 0, "is_drum": False, "genres": "pop"},
        {"path": str(song_b), "track": 0, "duration_seconds": 3.0, "program": 1, "is_drum": False, "genres": "jazz"},
    ])
    report = build_report(df)
    assert "summary" in report
    assert report["summary"]["n_songs"] == 2
    assert report["summary"]["median_song_duration_seconds"] == 2.0
