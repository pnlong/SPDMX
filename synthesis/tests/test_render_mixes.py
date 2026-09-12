"""Tests for ffmpeg full-song mix rendering."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from shared.config import SPDMX_AUDIO_DIR_NAME, SPDMX_FILE_NAME, SPDMX_MIX_DIR_NAME
from synthesis.render_mixes import (
    ffmpeg_sum_stems,
    render_dataset_mixes,
    update_stems_csv_mix_column,
)


def _write_mono_flac(path: Path, *, freq: float = 440.0, seconds: float = 0.05) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sr = 44100
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False, dtype=np.float32)
    sf.write(str(path), 0.1 * np.sin(2 * np.pi * freq * t), sr, format="FLAC")


@pytest.mark.skipif(
    __import__("shutil").which("ffmpeg") is None,
    reason="ffmpeg not on PATH",
)
def test_ffmpeg_amix_matches_python_sum(tmp_path: Path):
    """ffmpeg amix normalize=0 should match a raw float sum (within s16 FLAC)."""
    song = tmp_path / "audio" / "0/1/QmA"
    sr = 44100
    n = int(sr * 0.1)
    t = np.linspace(0, 0.1, n, endpoint=False, dtype=np.float32)
    # Amplitudes stay well below clip so s16 encode is the main error source.
    a = (0.2 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    b = (0.15 * np.sin(2 * np.pi * 554.0 * t)).astype(np.float32)
    c = (0.1 * np.cos(2 * np.pi * 660.0 * t)).astype(np.float32)
    paths = []
    for i, wave in enumerate((a, b, c)):
        path = song / f"{i}.flac"
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), wave, sr, format="FLAC")
        paths.append(path)

    dest = tmp_path / "mix" / "song.flac"
    assert ffmpeg_sum_stems(paths, dest) == "wrote"

    expected = a + b + c
    mix, mix_sr = sf.read(str(dest), always_2d=True, dtype="float32")
    assert mix_sr == sr
    assert mix.shape[1] == 1
    # Compare against raw Python sum (not peak-normalized).
    n_cmp = min(expected.shape[0], mix.shape[0])
    # s16 quantization ≈ 1/32768; allow a few LSBs + tiny amix drift.
    np.testing.assert_allclose(
        mix[:n_cmp, 0], expected[:n_cmp], rtol=1e-4, atol=2e-4,
    )
    assert abs(mix.shape[0] - expected.shape[0]) <= 1


@pytest.mark.skipif(
    __import__("shutil").which("ffmpeg") is None,
    reason="ffmpeg not on PATH",
)
def test_ffmpeg_sum_stems_writes_mono_and_skips(tmp_path: Path):
    song = tmp_path / "audio" / "0/1/QmA"
    _write_mono_flac(song / "0.flac", freq=440.0)
    _write_mono_flac(song / "1.flac", freq=554.0)
    dest = tmp_path / "mix" / "0/1/QmA.flac"
    assert ffmpeg_sum_stems([song / "0.flac", song / "1.flac"], dest) == "wrote"
    assert dest.is_file()
    info = sf.info(str(dest))
    assert info.channels == 1
    assert ffmpeg_sum_stems([song / "0.flac", song / "1.flac"], dest) == "skip_exists"
    assert ffmpeg_sum_stems([song / "0.flac", song / "1.flac"], dest, force=True) == "wrote"


@pytest.mark.skipif(
    __import__("shutil").which("ffmpeg") is None,
    reason="ffmpeg not on PATH",
)
def test_render_dataset_mixes_updates_csv(tmp_path: Path):
    root = tmp_path / "SPDMX_dev"
    song_id = "0/1/QmA"
    audio = root / SPDMX_AUDIO_DIR_NAME / song_id
    _write_mono_flac(audio / "0.flac")
    pd.DataFrame(
        [
            {
                "song_id": song_id,
                "path": f"./audio/{song_id}",
                "mid": f"./mid/{song_id}.mid",
                "track": 0,
                "original_track": 0,
                "program": 0,
                "is_drum": False,
                "name": "Piano",
            }
        ]
    ).to_csv(root / f"{SPDMX_FILE_NAME}.csv", index=False)

    counts = render_dataset_mixes(root, jobs=1, force=False)
    assert counts["wrote"] == 1
    mix_path = root / SPDMX_MIX_DIR_NAME / f"{song_id}.flac"
    assert mix_path.is_file()
    table = pd.read_csv(root / f"{SPDMX_FILE_NAME}.csv")
    assert table.iloc[0]["mix"] == f"./{SPDMX_MIX_DIR_NAME}/{song_id}.flac"

    counts2 = render_dataset_mixes(root, jobs=1, force=False)
    assert counts2["skip_exists"] == 1
    assert counts2.get("wrote", 0) == 0
    update_stems_csv_mix_column(root)


@pytest.mark.skipif(
    __import__("shutil").which("ffmpeg") is None,
    reason="ffmpeg not on PATH",
)
def test_render_dataset_mixes_resume_skips_without_reenqueue(tmp_path: Path, monkeypatch):
    """Parent resume scan must not call ffmpeg for songs that already have mixes."""
    root = tmp_path / "SPDMX_dev"
    song_id = "0/1/QmA"
    audio = root / SPDMX_AUDIO_DIR_NAME / song_id
    _write_mono_flac(audio / "0.flac")
    mix = root / SPDMX_MIX_DIR_NAME / f"{song_id}.flac"
    mix.parent.mkdir(parents=True, exist_ok=True)
    mix.write_bytes(b"already")
    pd.DataFrame(
        [
            {
                "song_id": song_id,
                "path": f"./audio/{song_id}",
                "mid": f"./mid/{song_id}.mid",
                "track": 0,
                "original_track": 0,
                "program": 0,
                "is_drum": False,
                "name": "Piano",
            }
        ]
    ).to_csv(root / f"{SPDMX_FILE_NAME}.csv", index=False)

    calls: list[object] = []

    def boom(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("ffmpeg_sum_stems should not run on resume")

    monkeypatch.setattr("synthesis.render_mixes.ffmpeg_sum_stems", boom)
    counts = render_dataset_mixes(root, jobs=1, force=False)
    assert counts["skip_exists"] == 1
    assert counts.get("wrote", 0) == 0
    assert calls == []
    assert mix.read_bytes() == b"already"


@pytest.mark.skipif(
    __import__("shutil").which("ffmpeg") is None,
    reason="ffmpeg not on PATH",
)
def test_render_dataset_mixes_reruns_when_stem_newer(tmp_path: Path):
    import os
    import time

    root = tmp_path / "SPDMX_dev"
    song_id = "0/1/QmA"
    audio = root / SPDMX_AUDIO_DIR_NAME / song_id
    _write_mono_flac(audio / "0.flac")
    mix = root / SPDMX_MIX_DIR_NAME / f"{song_id}.flac"
    mix.parent.mkdir(parents=True, exist_ok=True)
    mix.write_bytes(b"stale")
    older = time.time() - 120
    os.utime(mix, (older, older))
    newer = time.time()
    os.utime(audio / "0.flac", (newer, newer))
    pd.DataFrame(
        [
            {
                "song_id": song_id,
                "path": f"./audio/{song_id}",
                "mid": f"./mid/{song_id}.mid",
                "track": 0,
                "original_track": 0,
                "program": 0,
                "is_drum": False,
                "name": "Piano",
            }
        ]
    ).to_csv(root / f"{SPDMX_FILE_NAME}.csv", index=False)

    counts = render_dataset_mixes(root, jobs=1, force=False)
    assert counts["wrote"] == 1
    assert mix.is_file() and mix.stat().st_size > 10


@pytest.mark.skipif(
    __import__("shutil").which("ffmpeg") is None,
    reason="ffmpeg not on PATH",
)
def test_render_dataset_mixes_force_ids(tmp_path: Path):
    root = tmp_path / "SPDMX_dev"
    song_id = "0/1/QmA"
    audio = root / SPDMX_AUDIO_DIR_NAME / song_id
    _write_mono_flac(audio / "0.flac")
    pd.DataFrame(
        [
            {
                "song_id": song_id,
                "path": f"./audio/{song_id}",
                "mid": f"./mid/{song_id}.mid",
                "track": 0,
                "original_track": 0,
                "program": 0,
                "is_drum": False,
                "name": "Piano",
            }
        ]
    ).to_csv(root / f"{SPDMX_FILE_NAME}.csv", index=False)
    counts = render_dataset_mixes(root, jobs=1, force=False)
    assert counts["wrote"] == 1
    # Up to date → skip
    counts2 = render_dataset_mixes(root, jobs=1, force=False)
    assert counts2["skip_exists"] == 1
    # force_ids remakes even when fresh
    counts3 = render_dataset_mixes(root, jobs=1, force_ids={song_id})
    assert counts3["wrote"] == 1


@pytest.mark.skipif(
    __import__("shutil").which("ffmpeg") is None,
    reason="ffmpeg not on PATH",
)
def test_verify_mixes_match_stem_sums(tmp_path: Path):
    from synthesis.render_mixes import (
        ffmpeg_sum_stems,
        mix_matches_stem_sum,
        verify_mixes_match_stem_sums,
    )

    root = tmp_path / "SPDMX_dev"
    song_id = "0/1/QmSum"
    audio = root / SPDMX_AUDIO_DIR_NAME / song_id
    mix = root / SPDMX_MIX_DIR_NAME / f"{song_id}.flac"
    sr = 44100
    n = int(sr * 0.05)
    t = np.linspace(0, 0.05, n, endpoint=False, dtype=np.float32)
    a = (0.2 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    b = (0.15 * np.sin(2 * np.pi * 554.0 * t)).astype(np.float32)
    audio.mkdir(parents=True)
    sf.write(str(audio / "0.flac"), a, sr, format="FLAC")
    sf.write(str(audio / "1.flac"), b, sr, format="FLAC")
    assert ffmpeg_sum_stems([audio / "0.flac", audio / "1.flac"], mix) == "wrote"

    pd.DataFrame(
        [
            {
                "song_id": song_id,
                "path": f"./audio/{song_id}",
                "mid": f"./mid/{song_id}.mid",
                "track": 0,
                "original_track": 0,
                "program": 0,
                "is_drum": False,
                "name": "Piano",
            },
            {
                "song_id": song_id,
                "path": f"./audio/{song_id}",
                "mid": f"./mid/{song_id}.mid",
                "track": 1,
                "original_track": 1,
                "program": 0,
                "is_drum": False,
                "name": "Piano",
            },
        ]
    ).to_csv(root / f"{SPDMX_FILE_NAME}.csv", index=False)

    assert (
        mix_matches_stem_sum(
            song_id,
            audio_root=root / SPDMX_AUDIO_DIR_NAME,
            mix_root=root / SPDMX_MIX_DIR_NAME,
        )
        is None
    )
    verify_mixes_match_stem_sums(root, jobs=2)

    from synthesis.render_mixes import repair_mix_sum_mismatches

    # Corrupt mix → verify fails
    sf.write(str(mix), np.full(n, 0.99, np.float32), sr, format="FLAC")
    err = mix_matches_stem_sum(
        song_id,
        audio_root=root / SPDMX_AUDIO_DIR_NAME,
        mix_root=root / SPDMX_MIX_DIR_NAME,
    )
    assert err is not None and "max|mix-sum|" in err
    with pytest.raises(RuntimeError, match="sample-wise sum"):
        verify_mixes_match_stem_sums(root, jobs=1)

    # verify --delete-bad-mix-sums removes the bad file
    sf.write(str(mix), np.full(n, 0.99, np.float32), sr, format="FLAC")
    with pytest.raises(RuntimeError, match="deleted so mix can remake"):
        verify_mixes_match_stem_sums(root, jobs=1, delete_bad=True)
    assert not mix.is_file()

    # Repair = delete+remake via normal dirty path
    sf.write(str(mix), np.full(n, 0.99, np.float32), sr, format="FLAC")
    counts = repair_mix_sum_mismatches(root, jobs=1)
    assert counts["wrote"] == 1
    assert (
        mix_matches_stem_sum(
            song_id,
            audio_root=root / SPDMX_AUDIO_DIR_NAME,
            mix_root=root / SPDMX_MIX_DIR_NAME,
        )
        is None
    )
    verify_mixes_match_stem_sums(root, jobs=1)


@pytest.mark.skipif(
    __import__("shutil").which("ffmpeg") is None,
    reason="ffmpeg not on PATH",
)
def test_delete_bad_mixes_skips_non_content_and_preserves_good(tmp_path: Path):
    from synthesis.render_mixes import (
        delete_mix_sum_mismatches,
        ffmpeg_sum_stems,
        mix_sum_failure_is_safe_to_delete,
    )

    root = tmp_path / "SPDMX_dev"
    audio_root = root / SPDMX_AUDIO_DIR_NAME
    mix_root = root / SPDMX_MIX_DIR_NAME
    sr = 44100
    n = 1000
    wave = np.full(n, 0.1, np.float32)

    good_id = "0/0/QmGood"
    bad_id = "0/0/QmBad"
    nostem_id = "0/0/QmNoStem"

    for sid, corrupt in ((good_id, False), (bad_id, True)):
        audio = audio_root / sid
        audio.mkdir(parents=True)
        sf.write(str(audio / "0.flac"), wave, sr, format="FLAC")
        mix = mix_root / f"{sid}.flac"
        assert ffmpeg_sum_stems([audio / "0.flac"], mix) == "wrote"
        if corrupt:
            sf.write(str(mix), np.full(n, 0.99, np.float32), sr, format="FLAC")

    (mix_root / f"{nostem_id}.flac").parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(mix_root / f"{nostem_id}.flac"), wave, sr, format="FLAC")

    rows = []
    for sid in (good_id, bad_id, nostem_id):
        rows.append({
            "song_id": sid,
            "path": f"./audio/{sid}",
            "mid": f"./mid/{sid}.mid",
            "track": 0,
            "original_track": 0,
            "program": 0,
            "is_drum": False,
            "name": "Piano",
        })
    pd.DataFrame(rows).to_csv(root / f"{SPDMX_FILE_NAME}.csv", index=False)

    assert mix_sum_failure_is_safe_to_delete("x: max|mix-sum|=0.1")
    assert not mix_sum_failure_is_safe_to_delete("x: no audio stems")
    assert not mix_sum_failure_is_safe_to_delete("x: read failed (boom)")
    assert not mix_sum_failure_is_safe_to_delete("x: missing mix /tmp/x.flac")

    deleted = delete_mix_sum_mismatches(root, jobs=1)
    deleted_ids = {sid for sid, _ in deleted}
    assert deleted_ids == {bad_id}
    assert (mix_root / f"{good_id}.flac").is_file()
    assert (mix_root / f"{nostem_id}.flac").is_file()
    assert not (mix_root / f"{bad_id}.flac").is_file()


@pytest.mark.skipif(
    __import__("shutil").which("ffmpeg") is None,
    reason="ffmpeg not on PATH",
)
def test_delete_bad_mixes_refuses_large_fraction(tmp_path: Path):
    from synthesis.render_mixes import delete_mix_sum_mismatches, ffmpeg_sum_stems

    root = tmp_path / "SPDMX_dev"
    audio_root = root / SPDMX_AUDIO_DIR_NAME
    mix_root = root / SPDMX_MIX_DIR_NAME
    sr = 44100
    n = 500
    wave = np.full(n, 0.1, np.float32)
    rows = []
    for i in range(3):
        sid = f"0/0/Qm{i}"
        audio = audio_root / sid
        audio.mkdir(parents=True)
        sf.write(str(audio / "0.flac"), wave, sr, format="FLAC")
        mix = mix_root / f"{sid}.flac"
        assert ffmpeg_sum_stems([audio / "0.flac"], mix) == "wrote"
        sf.write(str(mix), np.full(n, 0.99, np.float32), sr, format="FLAC")
        rows.append({
            "song_id": sid,
            "path": f"./audio/{sid}",
            "mid": f"./mid/{sid}.mid",
            "track": 0,
            "original_track": 0,
            "program": 0,
            "is_drum": False,
            "name": "Piano",
        })
    pd.DataFrame(rows).to_csv(root / f"{SPDMX_FILE_NAME}.csv", index=False)

    with pytest.raises(RuntimeError, match="Refusing to delete"):
        delete_mix_sum_mismatches(root, jobs=1, force=False, max_count=2)
    for i in range(3):
        assert (mix_root / f"0/0/Qm{i}.flac").is_file()
    deleted = delete_mix_sum_mismatches(root, jobs=1, force=True, max_count=2)
    assert len(deleted) == 3
