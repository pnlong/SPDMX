"""Tests for PDMX song-length analysis."""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from analysis.analyze_song_lengths import build_report
from analysis.pdmx_lengths import load_pdmx_song_lengths, percentile_table
from analysis.plots import plot_histogram, plot_percentiles


def _write_fake_pdmx(path: Path, lengths: list[float]):
    pd.DataFrame({
        "subset:all_valid": [True] * len(lengths),
        "song_length.seconds": lengths,
    }).to_csv(path, index=False)


def test_load_pdmx_song_lengths(tmp_path: Path):
    csv_path = tmp_path / "PDMX.csv"
    _write_fake_pdmx(csv_path, [60.0, 200.0, 400.0])
    durations = load_pdmx_song_lengths(str(csv_path))
    assert len(durations) == 3
    assert list(durations) == [60.0, 200.0, 400.0]


def test_percentile_table():
    durations = pd.Series([10, 20, 30, 40, 100, 200, 300, 400, 500])
    table = percentile_table(durations)
    assert "p50" in table
    assert table["p50"] == 100.0


def test_build_report_has_summary_and_percentiles():
    durations = pd.Series([10, 50, 100, 130, 400])
    report = build_report(durations)
    assert "summary" in report
    assert "percentiles" in report
    assert report["n_songs"] == 5
    assert "sa3_limits" not in report
    assert "recommended_model" not in report["summary"]


def test_link_analysis_in_repo(tmp_path: Path, monkeypatch):
    from shared.repo_symlinks import link_analysis_in_repo
    from synthesis.paths import analysis_root, song_lengths_dir

    spdmx_root = str(tmp_path / "SPDMX")
    analysis_dir = Path(analysis_root(spdmx_root))
    song_lengths = Path(song_lengths_dir(spdmx_root))
    song_lengths.mkdir(parents=True)
    (song_lengths / "song_length_report.json").write_text("{}")

    symlink = tmp_path / "repo" / "analysis" / "output"
    symlink.parent.mkdir(parents=True)
    monkeypatch.setattr("shared.repo_symlinks.REPO_ANALYSIS_SYMLINK", symlink)
    monkeypatch.setattr("shared.repo_symlinks.LEGACY_ANALYSIS_SYMLINKS", ())

    link_analysis_in_repo(spdmx_root)
    assert symlink.is_symlink()
    assert symlink.resolve() == analysis_dir.resolve()
    assert (symlink / "song_lengths" / "song_length_report.json").exists()


def test_plots_write_files(tmp_path: Path):
    durations = pd.Series(np.linspace(10, 500, 100))
    histogram = tmp_path / "hist.png"
    percentiles = tmp_path / "pct.png"
    plot_histogram(durations, histogram)
    plot_percentiles(durations, percentiles)
    assert histogram.exists()
    assert percentiles.exists()
