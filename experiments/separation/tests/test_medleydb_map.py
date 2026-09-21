"""Unit tests for MedleyDB → BDGP stem mapping."""

from __future__ import annotations

from pathlib import Path

from experiments.separation.medleydb_map import (
    bdgp_stem_files,
    instrument_to_target,
    iter_medleydb_track_dirs,
    load_track_metadata,
)
from experiments.separation.paths import MEDLEYDB_METADATA_DIR, MEDLEYDB_ROOT


def test_instrument_to_target_basic():
    assert instrument_to_target("electric bass") == "bass"
    assert instrument_to_target("Drum Set") == "drums"
    assert instrument_to_target("distorted electric guitar") == "guitar"
    assert instrument_to_target("piano") == "piano"
    assert instrument_to_target("Main System") is None
    assert instrument_to_target("male singer") is None


def test_night_owl_bdgp_mapping():
    meta = load_track_metadata(MEDLEYDB_METADATA_DIR, "AClassicEducation_NightOwl")
    assert meta is not None
    files = bdgp_stem_files(meta)
    assert files["bass"] == ["AClassicEducation_NightOwl_STEM_01.wav"]
    assert files["drums"] == ["AClassicEducation_NightOwl_STEM_02.wav"]
    assert "AClassicEducation_NightOwl_STEM_03.wav" in files["guitar"]
    assert "piano" not in files


def test_iter_medleydb_v1_v2_if_present():
    if not MEDLEYDB_ROOT.is_dir():
        return
    tracks = list(iter_medleydb_track_dirs(MEDLEYDB_ROOT))
    assert len(tracks) >= 122
    ids = {t for t, _ in tracks}
    assert "AClassicEducation_NightOwl" in ids
    v2 = MEDLEYDB_ROOT / "V2"
    if v2.is_dir() and any(v2.iterdir()):
        assert len(tracks) >= 196
