"""Tests for transcription manifest helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from experiments.transcription.prepare_manifest import (
    _split_songs,
    hour_matched_subsample,
    write_arm_manifests,
)


def test_split_songs_cover_all():
    ids = [f"s{i}" for i in range(100)]
    splits = _split_songs(ids, seed=0, val_frac=0.1, test_frac=0.1)
    covered = set(splits["train"]) | set(splits["validation"]) | set(splits["test"])
    assert covered == set(ids)
    assert len(splits["test"]) >= 1
    assert len(splits["validation"]) >= 1


def test_hour_matched_and_write(tmp_path: Path):
    rows = []
    for i in range(20):
        rows.append(
            {
                "corpus": "spdmx",
                "split": "train",
                "song_id": f"t{i}",
                "mix_path": "",
                "midi_path": "",
                "n_stems": 2,
                "duration_sec": 3600.0,  # 1 hour each
            }
        )
    rows.append(
        {
            "corpus": "spdmx",
            "split": "test",
            "song_id": "hold",
            "mix_path": "",
            "midi_path": "",
            "n_stems": 2,
            "duration_sec": 100.0,
        }
    )
    df = pd.DataFrame(rows)
    sub = hour_matched_subsample(df, target_hours=5.0, seed=1)
    train_h = sub[sub["split"] == "train"]["duration_sec"].sum() / 3600.0
    assert 4.5 <= train_h <= 6.5
    assert (sub["split"] == "test").any()
    path = write_arm_manifests(sub, tmp_path, "spdmx_hour_matched")
    assert path.is_file()
