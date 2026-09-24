"""Tests for end-of-pipeline verification (patch only; preset_sweep removed)."""

from pathlib import Path

import pytest
import yaml

from experiments.listening.final_verify import (
    composed_config,
    verification_phase,
)
from experiments.listening_shared.clips import PRESET_SWEEP_REMOVED


def test_preset_sweep_removed():
    with pytest.raises(RuntimeError, match="preset_sweep"):
        verification_phase("preset")
    with pytest.raises(RuntimeError, match="preset_sweep"):
        composed_config("preset", "piano", "noise0.45")
    assert "preset_sweep" in PRESET_SWEEP_REMOVED


def _write_patch_winners(path: Path):
    doc = {
        "phases": {
            "phase1_soundfonts": {
                "completed": True,
                "winners": {"piano": "sgm"},
            },
            "phase2_fx": {
                "completed": True,
                "winners": {"piano": "dry"},
            },
        },
    }
    path.write_text(yaml.dump(doc))


def test_verification_phase_patch(tmp_path: Path):
    winners = tmp_path / "winners.yaml"
    _write_patch_winners(winners)
    assert verification_phase("patch", winners) == "phase1_soundfonts"


def test_composed_config_patch():
    assert composed_config("patch", "piano", "sgm") == {
        "variant_id": "sgm",
        "soundfont_id": "sgm",
        "fx_profile": "dry",
    }
