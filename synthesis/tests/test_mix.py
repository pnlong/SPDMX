"""Tests for the separate mixture + stem-normalization CLI."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch

from shared.config import DATA_DIR_NAME, SAMPLE_RATE
from synthesis.audio import load_stem, to_mono_numpy
from synthesis.mix import (
    confirm_overwrite,
    default_dest_dir,
    main,
    mix_command,
    normalize_stems_for_dataset,
    resolve_stems_dir,
)


def test_mix_command_includes_stems_dir_and_jobs():
    cmd = mix_command("/tmp/ablations/basic", jobs=8)
    assert cmd == "uv run python -m synthesis.mix --stems-dir /tmp/ablations/basic -j 8"


def test_mix_command_flac_flag():
    cmd = mix_command("/tmp/x", jobs=2, flac=True)
    assert "--flac" in cmd


def test_resolve_stems_dir_explicit(tmp_path: Path):
    assert resolve_stems_dir(stems_dir=str(tmp_path)) == tmp_path


def test_resolve_stems_dir_render_mode(tmp_path: Path):
    out = resolve_stems_dir(output_dir=str(tmp_path), render_mode="basic")
    assert out == tmp_path / "dev" / "ablations" / "basic"


def test_resolve_stems_dir_realify(tmp_path: Path):
    out = resolve_stems_dir(
        output_dir=str(tmp_path), render_mode="slakh", realify=True,
    )
    assert out == tmp_path / "dev" / "ablations" / "slakh_realify"


def test_default_dest_dir_sibling():
    assert default_dest_dir(Path("/a/basic")) == Path("/a/basic_summable")


def test_main_errors_on_missing_dir(tmp_path: Path):
    missing = tmp_path / "nope"
    with pytest.raises(SystemExit, match="Stem directory not found"):
        main(["--stems-dir", str(missing)])


def test_confirm_overwrite_yes(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "y")
    assert confirm_overwrite(Path("/tmp/x")) is True


def test_confirm_overwrite_no(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "n")
    assert confirm_overwrite(Path("/tmp/x")) is False


def test_main_aborts_without_confirm(tmp_path: Path, monkeypatch):
    stems_dir = tmp_path / "basic"
    song = stems_dir / "data" / "song"
    song.mkdir(parents=True)
    sf.write(str(song / "stem_0.flac"), np.zeros(100, np.float32), SAMPLE_RATE, format="FLAC")
    pd.DataFrame({"path": [str(song)], "track": [0]}).to_csv(stems_dir / "stems.csv", index=False)
    monkeypatch.setattr("synthesis.mix.confirm_overwrite", lambda _: False)
    with pytest.raises(SystemExit, match="Aborted"):
        main(["--stems-dir", str(stems_dir), "--flac"])


def _seed_song_tree(root: Path) -> Path:
    song = root / "data" / "song"
    song.mkdir(parents=True)
    sr = SAMPLE_RATE
    sf.write(str(song / "stem_0.flac"), np.full(sr, 0.9, np.float32), sr, format="FLAC")
    sf.write(str(song / "stem_1.flac"), np.full(sr, 0.9, np.float32), sr, format="FLAC")
    pd.DataFrame({
        "path": [str(song), str(song)],
        "track": [0, 1],
    }).to_csv(root / "stems.csv", index=False)
    pd.DataFrame({"path": [str(song)], "n_tracks": [2]}).to_csv(root / f"{DATA_DIR_NAME}.csv", index=False)
    return song


def test_no_overwrite_writes_dest_and_mixture(tmp_path: Path):
    source = tmp_path / "basic"
    song = _seed_song_tree(source)
    dest = tmp_path / "basic_summable"
    main([
        "--stems-dir", str(source),
        "--no-overwrite",
        "--dest-dir", str(dest),
        "--write-mixture",
        "--no-velocity-dynamics",
        "--flac",
        "-j", "1",
    ])
    # Originals untouched (still loud).
    orig0 = load_stem(song / "stem_0.flac")
    assert orig0.abs().max().item() > 0.8
    assert not (song / "mixture.flac").exists()

    out_song = dest / "data" / "song"
    stem0 = load_stem(out_song / "0.flac")
    stem1 = load_stem(out_song / "1.flac")
    mixture = load_stem(out_song / "mixture.flac")
    assert (stem0 + stem1).abs().max().item() <= 1.0 + 1e-4
    np.testing.assert_allclose(
        to_mono_numpy(stem0 + stem1), to_mono_numpy(mixture), rtol=1e-3, atol=1e-3,
    )
    remapped = pd.read_csv(dest / "stems.csv")
    assert str(remapped.iloc[0]["path"]).startswith(str(dest))


def test_overwrite_with_yes_skips_prompt(tmp_path: Path, monkeypatch):
    source = tmp_path / "basic"
    song = _seed_song_tree(source)
    called = {"n": 0}

    def boom(_):
        called["n"] += 1
        return False

    monkeypatch.setattr("synthesis.mix.confirm_overwrite", boom)
    main(["--stems-dir", str(source), "--flac", "--yes", "--no-velocity-dynamics", "-j", "1"])
    assert called["n"] == 0
    stem0 = load_stem(song / "stem_0.flac")
    stem1 = load_stem(song / "stem_1.flac")
    assert (stem0 + stem1).abs().max().item() <= 1.0 + 1e-4


def test_mix_resume_skips_complete_dest(tmp_path: Path):
    from synthesis.mix import build_mixture_tasks, mix_output_ready

    source = tmp_path / "raw"
    _seed_song_tree(source)
    # Canonical stem names under a separate dest (like raw → audio).
    dest_song = tmp_path / "audio" / "data" / "song"
    dest_song.mkdir(parents=True)
    for track in (0, 1):
        sf.write(
            str(dest_song / f"{track}.flac"),
            np.full(100, 0.1, np.float32),
            SAMPLE_RATE,
            format="FLAC",
        )
    assert mix_output_ready(dest_song, [0, 1], "flac")

    stems = pd.read_csv(source / "stems.csv")

    def to_audio(path: str) -> str:
        return str(path).replace(str(source), str(tmp_path / "audio"), 1)

    tasks, skipped = build_mixture_tasks(
        stems,
        source,
        source,
        "flac",
        write_mixture=False,
        use_velocity_dynamics=False,
        dest_song_dir_fn=to_audio,
        reset=False,
    )
    assert skipped == 1
    assert tasks == []

    tasks_reset, skipped_reset = build_mixture_tasks(
        stems,
        source,
        source,
        "flac",
        write_mixture=False,
        use_velocity_dynamics=False,
        dest_song_dir_fn=to_audio,
        reset=True,
    )
    assert skipped_reset == 0
    assert len(tasks_reset) == 1


def test_mix_resume_reruns_incomplete_dest(tmp_path: Path):
    from synthesis.mix import build_mixture_tasks

    source = tmp_path / "raw"
    _seed_song_tree(source)
    dest_song = tmp_path / "audio" / "data" / "song"
    dest_song.mkdir(parents=True)
    # Only track 0 present → must not skip.
    sf.write(
        str(dest_song / "0.flac"),
        np.full(100, 0.1, np.float32),
        SAMPLE_RATE,
        format="FLAC",
    )
    stems = pd.read_csv(source / "stems.csv")

    def to_audio(path: str) -> str:
        return str(path).replace(str(source), str(tmp_path / "audio"), 1)

    tasks, skipped = build_mixture_tasks(
        stems,
        source,
        source,
        "flac",
        use_velocity_dynamics=False,
        dest_song_dir_fn=to_audio,
    )
    assert skipped == 0
    assert len(tasks) == 1


def test_verify_mixed_stems_decodes_audio_tree(tmp_path: Path):
    from synthesis.audio import flac_fully_decodes
    from synthesis.mix import verify_mixed_stems_on_disk

    tables = tmp_path / "final"
    tables.mkdir()
    raw_song = tmp_path / "SPDMX" / "raw" / "1" / "2" / "QmX"
    audio_song = tmp_path / "SPDMX" / "audio" / "1" / "2" / "QmX"
    raw_song.mkdir(parents=True)
    audio_song.mkdir(parents=True)
    # stems.csv still points at raw/; verify_mix remaps to audio/
    pd.DataFrame({
        "path": [str(raw_song), str(raw_song)],
        "track": [0, 1],
    }).to_csv(tables / "stems.csv", index=False)
    for track in (0, 1):
        sf.write(
            str(audio_song / f"{track}.flac"),
            np.full(200, 0.05, np.float32),
            SAMPLE_RATE,
            format="FLAC",
        )
    assert flac_fully_decodes(audio_song / "0.flac")
    verify_mixed_stems_on_disk(tables, audio_format="flac", jobs=2)

    # Corrupt / empty → fail
    (audio_song / "1.flac").write_bytes(b"")
    with pytest.raises(RuntimeError, match="FLAC decode"):
        verify_mixed_stems_on_disk(tables, audio_format="flac", jobs=1)


def test_verify_mix_resolves_stale_paths_via_media_dir(tmp_path: Path):
    """Bookkeeping CSV may still say …/SPDMX/raw; media lives under SPDMX_dev."""
    from synthesis.mix import mixed_stem_paths_from_tables, verify_mixed_stems_on_disk

    tables = tmp_path / "final"
    media = tmp_path / "SPDMX_dev"
    tables.mkdir()
    stale_raw = tmp_path / "SPDMX" / "raw" / "1" / "2" / "QmX"
    audio_song = media / "audio" / "1" / "2" / "QmX"
    mix = media / "mix" / "1" / "2" / "QmX.flac"
    audio_song.mkdir(parents=True)
    mix.parent.mkdir(parents=True)
    pd.DataFrame({
        "path": [str(stale_raw), str(stale_raw)],
        "track": [0, 1],
    }).to_csv(tables / "stems.csv", index=False)
    # Production track map under media_dir (preferred).
    pd.DataFrame({
        "song_id": ["1/2/QmX", "1/2/QmX"],
        "path": ["./audio/1/2/QmX", "./audio/1/2/QmX"],
        "mid": ["./mid/1/2/QmX.mid", "./mid/1/2/QmX.mid"],
        "track": [0, 1],
        "original_track": [0, 1],
        "program": [0, 0],
        "is_drum": [False, False],
        "name": ["Piano", "Piano"],
    }).to_csv(media / "stems.csv", index=False)
    for track in (0, 1):
        sf.write(
            str(audio_song / f"{track}.flac"),
            np.full(200, 0.05, np.float32),
            SAMPLE_RATE,
            format="FLAC",
        )
    sf.write(str(mix), np.full((200, 2), 0.05, np.float32), SAMPLE_RATE, format="FLAC")

    paths = mixed_stem_paths_from_tables(tables, audio_format="flac", media_dir=media)
    assert all(str(media / "audio") in p for p in paths)
    assert not any("/SPDMX/audio/" in p.replace(str(media), "") for p in paths)
    verify_mixed_stems_on_disk(
        tables, audio_format="flac", jobs=1, media_dir=media,
    )
