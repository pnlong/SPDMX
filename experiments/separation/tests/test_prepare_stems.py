"""Tests for sPDMX pack indexing (songs.csv subset:bdgp filter)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd

from experiments.separation.prepare_stems import prepare_spdmx


def _write_stems(root: Path, rows: list[dict]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(root / "stems.csv", index=False)


def test_prepare_spdmx_filters_to_songs_csv_bdgp(tmp_path: Path):
    root = tmp_path / "SPDMX"
    out = tmp_path / "packs"
    # Two songs in stems; only song A is BDGP-flagged.
    _write_stems(
        root,
        [
            {"song_id": "A", "path": "./chunk_0/A", "chunk": 0, "track": 0,
             "program": 32, "is_drum": False},  # bass
            {"song_id": "A", "path": "./chunk_0/A", "chunk": 0, "track": 1,
             "program": 0, "is_drum": True},  # drums
            {"song_id": "A", "path": "./chunk_0/A", "chunk": 0, "track": 2,
             "program": 24, "is_drum": False},  # guitar
            {"song_id": "A", "path": "./chunk_0/A", "chunk": 0, "track": 3,
             "program": 0, "is_drum": False},  # piano
            {"song_id": "B", "path": "./chunk_0/B", "chunk": 0, "track": 0,
             "program": 0, "is_drum": False},  # piano only
        ],
    )
    pd.DataFrame(
        {
            "song_id": ["A", "B"],
            "subset:all": [True, True],
            "subset:bdgp": [True, False],
        }
    ).to_csv(root / "songs.csv", index=False)

    flac = tmp_path / "dummy.flac"
    flac.write_bytes(b"x")

    def fake_resolve(spdmx_root, row, *, song_id, track):
        return flac

    def fake_pool(job_list, fn, *, jobs_n, desc):
        return [{"song_id": j["song_id"]} for j in job_list]

    with (
        patch("experiments.separation.prepare_stems.resolve_spdmx_stem", fake_resolve),
        patch("experiments.separation.prepare_stems._run_pool", fake_pool),
    ):
        rows = prepare_spdmx(
            root, out, sample_rate=16000, require_all=True, jobs=1,
        )
    assert [r["song_id"] for r in rows] == ["A"]


def test_prepare_spdmx_falls_back_without_songs_csv(tmp_path: Path):
    root = tmp_path / "SPDMX"
    out = tmp_path / "packs"
    _write_stems(
        root,
        [
            {"song_id": "A", "path": "./audio/A", "track": 0, "program": 32, "is_drum": False},
            {"song_id": "A", "path": "./audio/A", "track": 1, "program": 0, "is_drum": True},
            {"song_id": "A", "path": "./audio/A", "track": 2, "program": 24, "is_drum": False},
            {"song_id": "A", "path": "./audio/A", "track": 3, "program": 0, "is_drum": False},
            {"song_id": "B", "path": "./audio/B", "track": 0, "program": 0, "is_drum": False},
        ],
    )
    flac = tmp_path / "dummy.flac"
    flac.write_bytes(b"x")

    def fake_resolve(spdmx_root, row, *, song_id, track):
        return flac

    def fake_pool(job_list, fn, *, jobs_n, desc):
        return [{"song_id": j["song_id"]} for j in job_list]

    with (
        patch("experiments.separation.prepare_stems.resolve_spdmx_stem", fake_resolve),
        patch("experiments.separation.prepare_stems._run_pool", fake_pool),
    ):
        rows = prepare_spdmx(
            root, out, sample_rate=16000, require_all=True, jobs=1,
        )
    assert [r["song_id"] for r in rows] == ["A"]
