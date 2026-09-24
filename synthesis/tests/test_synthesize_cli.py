"""Tests for synthesize CLI orchestration."""

from pathlib import Path

import pandas as pd
import pytest

from shared.config import DATA_DIR_NAME, DEFAULT_AUDIO_FORMAT, STEMS_FILE_NAME
from synthesis.synthesize import (
    load_completed_song_paths,
    require_raw_synthesis,
    reset_synthesis_output,
    songs_missing_routing,
    synthesis_is_complete,
)


def _write_complete_ablation(ablation_dir: Path, n_tracks: int = 1):
    ablation_dir.mkdir(parents=True)
    song_dir = ablation_dir / "data" / "song"
    song_dir.mkdir(parents=True)
    import numpy as np
    import soundfile as sf

    sr = 44100
    for j in range(n_tracks):
        sf.write(str(song_dir / f"stem_{j}.mp3"), np.zeros(sr), sr, format="MP3")
    sf.write(str(song_dir / "mixture.mp3"), np.zeros(sr), sr, format="MP3")

    pd.DataFrame({"path": [str(song_dir)], "n_tracks": [n_tracks]}).to_csv(
        ablation_dir / f"{DATA_DIR_NAME}.csv", index=False
    )
    pd.DataFrame({
        "path": [str(song_dir)] * n_tracks,
        "track": list(range(n_tracks)),
    }).to_csv(ablation_dir / f"{STEMS_FILE_NAME}.csv", index=False)


def test_synthesis_is_complete_false_when_stems_missing(tmp_path: Path):
    ablation_dir = tmp_path / "basic"
    ablation_dir.mkdir()
    song_dir = ablation_dir / "data" / "song"
    song_dir.mkdir(parents=True)
    pd.DataFrame({"path": [str(song_dir)], "n_tracks": [1]}).to_csv(
        ablation_dir / f"{DATA_DIR_NAME}.csv", index=False
    )
    pd.DataFrame({"path": [str(song_dir)], "track": [0]}).to_csv(
        ablation_dir / f"{STEMS_FILE_NAME}.csv", index=False
    )
    assert not synthesis_is_complete(str(ablation_dir), DEFAULT_AUDIO_FORMAT)


def test_require_raw_synthesis_raises_with_command(tmp_path: Path):
    ablation_dir = tmp_path / "basic"
    ablation_dir.mkdir()
    with pytest.raises(RuntimeError, match="Run the corresponding ablation first"):
        require_raw_synthesis(
            str(ablation_dir),
            run_command="uv run python -m synthesis.synthesize --render-mode basic",
        )


def test_require_raw_synthesis_passes_when_complete(tmp_path: Path):
    ablation_dir = tmp_path / "basic"
    _write_complete_ablation(ablation_dir)
    require_raw_synthesis(
        str(ablation_dir),
        run_command="uv run python -m synthesis.synthesize --render-mode basic",
    )


def test_synthesis_is_complete_stems_only_without_mixture(tmp_path: Path):
    ablation_dir = tmp_path / "basic"
    ablation_dir.mkdir()
    song_dir = ablation_dir / "data" / "song"
    song_dir.mkdir(parents=True)
    import numpy as np
    import soundfile as sf

    sf.write(str(song_dir / "stem_0.mp3"), np.zeros(44100), 44100, format="MP3")
    pd.DataFrame({"path": [str(song_dir)], "n_tracks": [1]}).to_csv(
        ablation_dir / f"{DATA_DIR_NAME}.csv", index=False
    )
    pd.DataFrame({"path": [str(song_dir)], "track": [0]}).to_csv(
        ablation_dir / f"{STEMS_FILE_NAME}.csv", index=False
    )
    # Synthesis completeness is stems-only; mixtures are a separate mix pass.
    assert synthesis_is_complete(str(ablation_dir), DEFAULT_AUDIO_FORMAT)
    assert not synthesis_is_complete(
        str(ablation_dir), DEFAULT_AUDIO_FORMAT, require_mixture=True,
    )


def test_reset_synthesis_output_removes_stems_and_tables(tmp_path: Path):
    ablation_dir = tmp_path / "basic"
    _write_complete_ablation(ablation_dir)
    assert (ablation_dir / f"{DATA_DIR_NAME}.csv").exists()
    assert (ablation_dir / "data" / "song" / "stem_0.mp3").exists()

    reset_synthesis_output(str(ablation_dir))

    assert ablation_dir.is_dir()
    assert not (ablation_dir / f"{DATA_DIR_NAME}.csv").exists()
    assert not (ablation_dir / "data" / "song" / "stem_0.mp3").exists()


def test_songs_missing_routing_detects_incomplete_coverage():
    songs = pd.DataFrame({
        "path": ["/a", "/b", "/c"],
        "n_tracks": [2, 1, 1],
    })
    routing = pd.DataFrame({
        "path": ["/a", "/a", "/c", "/c"],
        "track": [0, 1, 0, 1],
    })
    # /b missing entirely; /c has extra routing tracks but covers 0..n_tracks-1.
    assert songs_missing_routing(songs, routing) == {"/b"}
    assert songs_missing_routing(songs, pd.DataFrame()) == {"/a", "/b", "/c"}


def test_load_completed_song_paths_excludes_incomplete_routing(tmp_path: Path):
    data_csv = tmp_path / f"{DATA_DIR_NAME}.csv"
    routing_csv = tmp_path / "ddsp_routing.csv"
    pd.DataFrame({
        "path": ["/done", "/missing"],
        "n_tracks": [1, 2],
    }).to_csv(data_csv, index=False)
    pd.DataFrame({
        "path": ["/done"],
        "track": [0],
    }).to_csv(routing_csv, index=False)

    assert load_completed_song_paths(data_csv) == {"/done", "/missing"}
    assert load_completed_song_paths(data_csv, routing_csv=routing_csv) == {"/done"}


def test_synthesis_is_complete_requires_ddsp_routing_when_present(tmp_path: Path):
    ablation_dir = tmp_path / "ddsp_basic"
    _write_complete_ablation(ablation_dir)
    # Routing file present but empty coverage → incomplete.
    pd.DataFrame(columns=["path", "track"]).to_csv(
        ablation_dir / "ddsp_routing.csv", index=False,
    )
    assert not synthesis_is_complete(str(ablation_dir), DEFAULT_AUDIO_FORMAT)

    song_dir = ablation_dir / "data" / "song"
    pd.DataFrame({"path": [str(song_dir)], "track": [0]}).to_csv(
        ablation_dir / "ddsp_routing.csv", index=False,
    )
    assert synthesis_is_complete(str(ablation_dir), DEFAULT_AUDIO_FORMAT)


def test_synthesis_is_complete_expected_n_songs(tmp_path: Path):
    ablation_dir = tmp_path / "basic"
    _write_complete_ablation(ablation_dir)
    assert synthesis_is_complete(str(ablation_dir), DEFAULT_AUDIO_FORMAT)
    assert not synthesis_is_complete(
        str(ablation_dir), DEFAULT_AUDIO_FORMAT, expected_n_songs=2,
    )
    assert synthesis_is_complete(
        str(ablation_dir), DEFAULT_AUDIO_FORMAT, expected_n_songs=1,
    )
