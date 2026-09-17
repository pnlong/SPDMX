"""Smoke tests for StreamGen index builder helpers."""

from __future__ import annotations

from experiments.transcription.prepare_manifest import _split_songs


def test_split_reusable():
    splits = _split_songs([f"x{i}" for i in range(40)], seed=2, val_frac=0.1, test_frac=0.1)
    assert len(splits["train"]) + len(splits["validation"]) + len(splits["test"]) == 40
