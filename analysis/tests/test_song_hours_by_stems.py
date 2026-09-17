"""Unit tests for song-hours-by-stems aggregation helpers."""

from __future__ import annotations

from analysis.song_hours_by_stems import _bin_hours


def test_bin_hours_overflow():
    raw = {2: 10.0, 3: 5.0, 25: 2.0, 30: 1.0}
    binned = _bin_hours(raw, max_bin=20)
    assert binned["labels"][0] == "2"
    assert binned["labels"][-1] == "20+"
    assert abs(binned["hours"][0] - 10.0) < 1e-6
    assert abs(binned["hours"][-1] - 3.0) < 1e-6
