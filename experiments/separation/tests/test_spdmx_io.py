"""Tests for chunked / flat sPDMX stem path resolution."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from experiments.separation.spdmx_io import resolve_spdmx_stem


def test_resolve_prefers_path_column(tmp_path: Path):
    root = tmp_path / "SPDMX"
    stem = root / "chunk_0" / "audio" / "0/1/QmA" / "0.flac"
    stem.parent.mkdir(parents=True)
    stem.write_bytes(b"flac")
    row = pd.Series(
        {
            "path": "./chunk_0/audio/0/1/QmA",
            "chunk": 0,
            "track": 0,
        }
    )
    assert resolve_spdmx_stem(root, row, song_id="0/1/QmA", track=0) == stem


def test_resolve_uses_chunk_column(tmp_path: Path):
    root = tmp_path / "SPDMX"
    stem = root / "chunk_2" / "audio" / "1/2/QmB" / "1.flac"
    stem.parent.mkdir(parents=True)
    stem.write_bytes(b"flac")
    row = pd.Series({"chunk": 2, "track": 1})
    assert resolve_spdmx_stem(root, row, song_id="1/2/QmB", track=1) == stem


def test_resolve_falls_back_to_flat_audio(tmp_path: Path):
    root = tmp_path / "SPDMX_dev"
    stem = root / "audio" / "0/1/QmA" / "0.flac"
    stem.parent.mkdir(parents=True)
    stem.write_bytes(b"flac")
    row = pd.Series({"path": "./audio/0/1/QmA", "track": 0})
    assert resolve_spdmx_stem(root, row, song_id="0/1/QmA", track=0) == stem
