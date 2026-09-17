"""Tests for the StreamGen SPDMX dataset adapter (no audio I/O)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

# Import from the versioned copy in experiments/ (not the upstream clone).
import importlib.util


def _load_spdmx_module():
    path = (
        Path(__file__).resolve().parents[1] / "adapters" / "spdmx.py"
    )
    # Stub stream_music_gen.constants if missing in test env path tricks —
    # tests run with repo root on path and upstream installed editable.
    spec = importlib.util.spec_from_file_location("spdmx_adapter_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_gm_program_mapping():
    mod = _load_spdmx_module()
    # Piano (0) → Piano class; drums flag → Drums
    piano = mod._gm_program_to_class_id(0, False)
    drums = mod._gm_program_to_class_id(0, True)
    assert piano != drums
    assert drums == mod.inst_name_to_inst_class_id("Drums")


def test_metadata_from_jsonl(tmp_path: Path):
    mod = _load_spdmx_module()
    audio_root = tmp_path / "SPDMX"
    song = audio_root / "chunk_0" / "1" / "11" / "QmTest"
    song.mkdir(parents=True)
    stem = song / "0.flac"
    stem.write_bytes(b"fake")  # not loaded in metadata-only path

    index = tmp_path / "spdmx" / "spdmx_multitrack.jsonl"
    index.parent.mkdir(parents=True)
    rec = {
        "corpus": "spdmx",
        "split": "train",
        "song_id": "1/11/QmTest",
        "mix_path": str(song / "mix.flac"),
        "n_stems": 1,
        "duration_sec": 1.0,
        "stems": [
            {
                "track": 0,
                "program": 0,
                "is_drum": False,
                "audio_path": str(stem),
            }
        ],
    }
    index.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    (tmp_path / "spdmx" / "audio").symlink_to(audio_root)

    ds = mod.Spdmx(
        root_dir=str(tmp_path / "spdmx"),
        split="train",
        target_sample_rate=32000,
        regenerate_metadata=True,
        download=False,
    )
    assert len(ds) == 1
    row = ds.all_metadata.iloc[0]
    assert row["track_name"] == "1/11/QmTest"
    assert row["audio_path"] == "chunk_0/1/11/QmTest/0.flac"
    assert row["instrument_class_id"] == mod._gm_program_to_class_id(0, False)
    assert (tmp_path / "spdmx" / "pt_dataset_metadata_train.parquet").is_file()
