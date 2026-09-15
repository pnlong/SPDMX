"""Tests for RMS-gated active hours helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from analysis.active_hours import (
    active_seconds_from_mono,
    aggregate_hours_by_program,
    gm_id_from_stem_row,
    measure_stem_active_seconds,
)
from analysis.gm_programs import DRUM_GM_ID
from analysis.plots import plot_instrument_active_hours


def test_active_seconds_all_silent():
    sr = 16000
    mono = np.zeros(sr, dtype=np.float32)
    wall, active = active_seconds_from_mono(mono, sample_rate=sr, min_rms=0.01)
    assert wall == 1.0
    assert active == 0.0


def test_active_seconds_half_active():
    sr = 16000
    mono = np.zeros(2 * sr, dtype=np.float32)
    mono[sr:] = 0.2
    wall, active = active_seconds_from_mono(
        mono, sample_rate=sr, hop_seconds=0.25, min_rms=0.01
    )
    assert abs(wall - 2.0) < 1e-6
    assert 0.9 < active < 1.1


def test_gm_id_from_stem_row():
    assert gm_id_from_stem_row(0, False) == 0
    assert gm_id_from_stem_row(0, True) == DRUM_GM_ID


def test_measure_and_aggregate(tmp_path: Path):
    sr = 8000
    path = tmp_path / "0.flac"
    audio = np.zeros((sr, 1), dtype=np.float32)
    audio[sr // 2 :, 0] = 0.2
    sf.write(path, audio, sr)

    measured = measure_stem_active_seconds(path, hop_seconds=0.25, min_rms=0.01)
    assert measured is not None
    wall, active = measured
    assert abs(wall - 1.0) < 1e-3
    assert 0.4 < active < 0.6

    rows = pd.DataFrame(
        [
            {"gm_id": 0, "wall_seconds": wall, "active_seconds": active},
            {"gm_id": 0, "wall_seconds": 2.0, "active_seconds": 1.0},
            {"gm_id": DRUM_GM_ID, "wall_seconds": 3.0, "active_seconds": 1.5},
        ]
    )
    summary = aggregate_hours_by_program(rows)
    assert list(summary["gm_id"]) == [0, DRUM_GM_ID]
    out = tmp_path / "hours.pdf"
    plot_instrument_active_hours(summary, out, top_n=5)
    assert out.exists()
