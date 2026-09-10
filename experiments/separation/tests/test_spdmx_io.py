"""Tests for chunked / flat sPDMX stem path resolution."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from experiments.separation.spdmx_io import resolve_spdmx_mix, resolve_spdmx_stem


def test_resolve_prefers_path_column_flattened(tmp_path: Path):
    root = tmp_path / "SPDMX"
    stem = root / "chunk_0" / "0/1/QmA" / "0.flac"
    stem.parent.mkdir(parents=True)
    stem.write_bytes(b"flac")
    row = pd.Series(
        {
            "path": "./chunk_0/0/1/QmA",
            "chunk": 0,
            "track": 0,
        }
    )
    assert resolve_spdmx_stem(root, row, song_id="0/1/QmA", track=0) == stem


def test_resolve_uses_chunk_column_flattened(tmp_path: Path):
    root = tmp_path / "SPDMX"
    stem = root / "chunk_2" / "1/2/QmB" / "1.flac"
    stem.parent.mkdir(parents=True)
    stem.write_bytes(b"flac")
    row = pd.Series({"chunk": 2, "track": 1})
    assert resolve_spdmx_stem(root, row, song_id="1/2/QmB", track=1) == stem


def test_resolve_flat_audio(tmp_path: Path):
    root = tmp_path / "SPDMX_dev"
    stem = root / "audio" / "0/1/QmA" / "0.flac"
    stem.parent.mkdir(parents=True)
    stem.write_bytes(b"flac")
    row = pd.Series({"path": "./audio/0/1/QmA", "track": 0})
    assert resolve_spdmx_stem(root, row, song_id="0/1/QmA", track=0) == stem


def test_resolve_mix_flattened_and_flat(tmp_path: Path):
    release = tmp_path / "SPDMX"
    mix = release / "chunk_0" / "0/1/QmA" / "mix.flac"
    mix.parent.mkdir(parents=True)
    mix.write_bytes(b"mix")
    row = pd.Series(
        {
            "path": "./chunk_0/0/1/QmA",
            "mix": "./chunk_0/0/1/QmA/mix.flac",
            "chunk": 0,
        }
    )
    assert resolve_spdmx_mix(release, row, song_id="0/1/QmA") == mix

    flat = tmp_path / "SPDMX_dev"
    flat_mix = flat / "mix" / "0/1/QmA.flac"
    flat_mix.parent.mkdir(parents=True)
    flat_mix.write_bytes(b"mix")
    assert resolve_spdmx_mix(flat, pd.Series(dtype=object), song_id="0/1/QmA") == flat_mix
