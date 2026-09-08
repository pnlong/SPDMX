"""Unit tests for GM / Slakh target mapping and SI-SDR."""

from __future__ import annotations

import numpy as np

from experiments.separation.sisdr import si_sdr
from experiments.separation.targets import gm_to_target, slakh_inst_class_to_target


def test_gm_to_target_ranges():
    assert gm_to_target(0, False) == "piano"
    assert gm_to_target(7, False) == "piano"
    assert gm_to_target(8, False) is None
    assert gm_to_target(24, False) == "guitar"
    assert gm_to_target(31, False) == "guitar"
    assert gm_to_target(32, False) == "bass"
    assert gm_to_target(39, False) == "bass"
    assert gm_to_target(40, False) is None
    assert gm_to_target(0, True) == "drums"


def test_slakh_inst_class():
    assert slakh_inst_class_to_target("Bass") == "bass"
    assert slakh_inst_class_to_target("Drums") == "drums"
    assert slakh_inst_class_to_target("Guitar") == "guitar"
    assert slakh_inst_class_to_target("Piano") == "piano"
    assert slakh_inst_class_to_target("Strings (continued)") is None
    assert slakh_inst_class_to_target(None, is_drum=True) == "drums"


def test_si_sdr_identical():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(8000)
    assert si_sdr(x, x) > 50


def test_si_sdr_scaled():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(8000)
    assert si_sdr(2.5 * x, x) > 50
