"""Tests for multistem stem/other dataset helpers and val crop selection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiments.separation.audio_io import write_flac
from experiments.separation.dataset_multistem import (
    StemOtherDataset,
    energy_ratio,
    val_crop_starts,
)


def test_val_crop_starts_short():
    assert val_crop_starts(100, 200) == [0]


def test_val_crop_starts_grid():
    starts = val_crop_starts(1000, 100)
    assert starts[0] == 0
    assert starts[-1] == 900
    assert 0.25 * 900 in starts or any(abs(s - 225) <= 1 for s in starts)


def test_energy_ratio_silent_mix():
    mix = np.zeros(64, dtype=np.float32)
    stem = np.ones(64, dtype=np.float32)
    assert energy_ratio(mix, stem) == 0.0


def _write_song(root: Path, song_rel: str, stems: dict[int, np.ndarray], sr: int = 8000) -> None:
    song_dir = root / song_rel
    song_dir.mkdir(parents=True)
    for track, audio in stems.items():
        write_flac(song_dir / f"{track}.flac", audio, sr)
    # Deliberately wrong mix.flac (should be ignored when rebuild_mix_from_stems).
    write_flac(song_dir / "mix.flac", np.ones(stems[0].shape[0], dtype=np.float32), sr)


def test_val_picks_energetic_crop_not_silent_intro(tmp_path: Path):
    root = tmp_path / "SPDMX"
    sr = 8000
    seg = 1.0
    n = int(4 * sr)  # 4 seconds
    silent = np.zeros(n, dtype=np.float32)
    # Stem 0: energy only in the second half; stem 1: quiet everywhere.
    stem0 = silent.copy()
    stem0[2 * sr :] = 0.4
    stem1 = silent.copy()
    stem1[:] = 0.01
    # Mix ≈ stem0 + stem1 when rebuilt.
    _write_song(root, "chunk_0/0/1/QmTest", {0: stem0, 1: stem1}, sr=sr)

    manifest = tmp_path / "val.csv"
    pd.DataFrame(
        [
            {
                "song_id": "0/1/QmTest",
                "path": "./chunk_0/0/1/QmTest",
                "mix": "chunk_0/0/1/QmTest/mix.flac",
                "tracks": "0|1",
            }
        ]
    ).to_csv(manifest, index=False)

    ds = StemOtherDataset(
        manifest,
        root,
        sample_rate=sr,
        segment_seconds=seg,
        channels=1,
        train=False,
        stem_min_energy_ratio=0.05,
        rebuild_mix_from_stems=True,
    )
    batch = ds[0]
    mix = batch["mix"].numpy().reshape(-1)
    stem = batch["sources"][0].numpy().reshape(-1)
    # Should not be the silent first second of stem 0.
    assert float(np.sqrt(np.mean(stem**2))) > 0.1
    assert energy_ratio(mix, stem) >= 0.05
    # other + stem ≈ mix
    other = batch["sources"][1].numpy().reshape(-1)
    assert torch.allclose(
        torch.from_numpy(mix),
        torch.from_numpy(stem + other),
        atol=1e-5,
    )


def test_rebuild_mix_ignores_bad_mix_flac(tmp_path: Path):
    root = tmp_path / "SPDMX"
    sr = 8000
    n = sr  # 1 s segment
    a = np.full(n, 0.2, dtype=np.float32)
    b = np.full(n, 0.3, dtype=np.float32)
    _write_song(root, "chunk_0/0/1/QmSum", {0: a, 1: b}, sr=sr)

    manifest = tmp_path / "train.csv"
    pd.DataFrame(
        [
            {
                "song_id": "0/1/QmSum",
                "path": "./chunk_0/0/1/QmSum",
                "mix": "chunk_0/0/1/QmSum/mix.flac",
                "tracks": "0|1",
            }
        ]
    ).to_csv(manifest, index=False)

    ds = StemOtherDataset(
        manifest,
        root,
        sample_rate=sr,
        segment_seconds=1.0,
        channels=1,
        train=False,
        stem_min_energy_ratio=0.0,
        rebuild_mix_from_stems=True,
    )
    batch = ds[0]
    mix = batch["mix"].numpy().reshape(-1)
    # Sum of stems is 0.5, not the bogus ones mix.flac.
    assert abs(float(mix.mean()) - 0.5) < 1e-3
