"""Tests for post-render chunk assignment and packaging."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd
import pytest

from shared.config import SPDMX_AUDIO_DIR_NAME, SPDMX_FILE_NAME, SPDMX_MID_DIR_NAME
from synthesis.build_spdmx import chunk_dataset
from synthesis.chunking import (
    CHUNKS_FILE_NAME,
    PACKAGED_TRACK_MAP_COLUMNS,
    SHA256SUMS_FILE_NAME,
    assign_songs_to_chunks,
    build_chunks_manifest,
    chunk_dir_name,
    format_chunk_id,
    list_chunk_dirs,
    packaged_audio_rel,
    packaged_mid_rel,
    rewrite_track_map_for_chunks,
)
from synthesis.distribute_spdmx import stage_zenodo_files
from synthesis.spdmx_release import write_spdmx_release_docs


def test_format_chunk_id_and_dir_name():
    assert format_chunk_id(0) == "0"
    assert format_chunk_id(12) == "12"
    assert chunk_dir_name("0") == "chunk_0"
    assert chunk_dir_name("chunk_7") == "chunk_7"
    assert chunk_dir_name("007") == "chunk_7"


def test_assign_songs_to_chunks_respects_budget_and_seed():
    sizes = {
        "a/1/QmA": 10,
        "b/1/QmB": 10,
        "c/1/QmC": 10,
        "d/1/QmD": 5,
    }
    first = assign_songs_to_chunks(sizes, target_bytes=20, seed=1)
    second = assign_songs_to_chunks(sizes, target_bytes=20, seed=1)
    assert first == second
    assert set(first) == set(sizes)
    # Every song lands in some chunk; packing should use more than one chunk.
    assert len(set(first.values())) >= 2
    # Bytes per chunk never exceed budget except for a lone oversized song.
    for chunk_id in set(first.values()):
        total = sum(sizes[s] for s, c in first.items() if c == chunk_id)
        songs = [s for s, c in first.items() if c == chunk_id]
        if len(songs) == 1 and sizes[songs[0]] > 20:
            assert total == sizes[songs[0]]
        else:
            assert total <= 20


def test_oversized_song_gets_own_chunk():
    sizes = {"big": 100, "small": 5}
    assignment = assign_songs_to_chunks(sizes, target_bytes=20, seed=0)
    assert len({c for s, c in assignment.items() if s == "big"}) == 1
    # big alone in its chunk
    big_chunk = assignment["big"]
    assert [s for s, c in assignment.items() if c == big_chunk] == ["big"]


def test_rewrite_track_map_for_chunks_updates_paths():
    table = pd.DataFrame(
        [
            {
                "song_id": "0/1/QmA",
                "path": "./audio/0/1/QmA",
                "mid": "./mid/0/1/QmA.mid",
                "track": 0,
                "original_track": 0,
                "program": 0,
                "is_drum": False,
                "name": "Piano",
            },
            {
                "song_id": "0/1/QmA",
                "path": "./audio/0/1/QmA",
                "mid": "./mid/0/1/QmA.mid",
                "track": 1,
                "original_track": 2,
                "program": 40,
                "is_drum": False,
                "name": "Violin",
            },
        ]
    )
    assignment = {"0/1/QmA": "0"}
    packaged = rewrite_track_map_for_chunks(table, assignment)
    assert list(packaged.columns[: len(PACKAGED_TRACK_MAP_COLUMNS)]) == (
        PACKAGED_TRACK_MAP_COLUMNS
    )
    assert (packaged["chunk"] == "0").all()
    assert packaged.iloc[0]["path"] == packaged_audio_rel("0", "0/1/QmA")
    assert packaged.iloc[0]["mid"] == packaged_mid_rel("0", "0/1/QmA")


def test_build_chunks_manifest_counts():
    packaged = pd.DataFrame(
        {
            "song_id": ["a", "a", "b"],
            "chunk": ["0", "0", "1"],
            "path": [".", ".", "."],
            "mid": [".", ".", "."],
            "track": [0, 1, 0],
            "original_track": [0, 1, 0],
            "program": [0, 0, 0],
            "is_drum": [False, False, False],
            "name": ["x", "y", "z"],
        }
    )
    assignment = {"a": "0", "b": "1"}
    sizes = {"a": 10, "b": 20}
    manifest = build_chunks_manifest(packaged, assignment, sizes)
    assert list(manifest["chunk"]) == ["0", "1"]
    assert list(manifest["n_songs"]) == [1, 1]
    assert list(manifest["n_stems"]) == [2, 1]
    assert list(manifest["bytes"]) == [10, 20]
    assert list(manifest["archive"]) == ["", ""]


def _write_flat_fixture(root: Path) -> None:
    rows = []
    for song_id, tracks in (("0/1/QmA", 2), ("0/2/QmB", 1)):
        audio = root / SPDMX_AUDIO_DIR_NAME / song_id
        audio.mkdir(parents=True)
        for track in range(tracks):
            (audio / f"{track}.flac").write_bytes(b"flac" * (20 if song_id.endswith("A") else 5))
            rows.append(
                {
                    "song_id": song_id,
                    "path": f"./audio/{song_id}",
                    "mid": f"./mid/{song_id}.mid",
                    "track": track,
                    "original_track": track,
                    "program": 0,
                    "is_drum": False,
                    "name": "Piano",
                }
            )
        mid = root / SPDMX_MID_DIR_NAME / f"{song_id}.mid"
        mid.parent.mkdir(parents=True, exist_ok=True)
        mid.write_bytes(b"MThd")
    pd.DataFrame(rows).to_csv(root / f"{SPDMX_FILE_NAME}.csv", index=False)
    write_spdmx_release_docs(root)


def test_publish_dir_leaves_source_if_verify_would_fail(tmp_path: Path, monkeypatch):
    from synthesis import build_spdmx as mod

    src = tmp_path / "audio" / "0/1/QmA"
    dst = tmp_path / "chunk_0" / "audio" / "0/1/QmA"
    src.mkdir(parents=True)
    (src / "0.flac").write_bytes(b"abc")

    def bad_mirror(s, d, *, copy):
        d.mkdir(parents=True, exist_ok=True)
        (d / "0.flac").write_bytes(b"nope")

    monkeypatch.setattr(mod, "_mirror_files", bad_mirror)
    with pytest.raises(RuntimeError, match="verify failed"):
        mod._publish_dir(src, dst, copy=False)
    assert (src / "0.flac").read_bytes() == b"abc"


def test_chunk_dataset_refuses_in_place_by_default(tmp_path: Path):
    source = tmp_path / "SPDMX_dev"
    source.mkdir()
    _write_flat_fixture(source)
    with pytest.raises(ValueError, match="in-place"):
        chunk_dataset(dataset_dir=source, package_dir=source, target_bytes=50)


def test_chunk_dataset_builds_separate_release_tree(tmp_path: Path):
    source = tmp_path / "SPDMX_dev"
    dest = tmp_path / "SPDMX"
    source.mkdir()
    _write_flat_fixture(source)

    packaged, chunks, assignment = chunk_dataset(
        dataset_dir=source,
        package_dir=dest,
        target_bytes=50,
        seed=0,
    )
    assert set(assignment) == {"0/1/QmA", "0/2/QmB"}
    assert "chunk" in packaged.columns
    assert (dest / f"{SPDMX_FILE_NAME}.csv").is_file()
    assert (dest / CHUNKS_FILE_NAME).is_file()
    assert (dest / "LICENSE").is_file()

    # Flat production tree untouched.
    assert (source / SPDMX_AUDIO_DIR_NAME / "0/1/QmA" / "0.flac").is_file()
    assert (source / SPDMX_MID_DIR_NAME / "0/1/QmA.mid").is_file()
    assert "chunk" not in pd.read_csv(source / f"{SPDMX_FILE_NAME}.csv").columns

    for song_id, chunk_id in assignment.items():
        assert (
            dest
            / chunk_dir_name(chunk_id)
            / SPDMX_AUDIO_DIR_NAME
            / song_id
            / "0.flac"
        ).is_file()
        assert (
            dest
            / chunk_dir_name(chunk_id)
            / SPDMX_MID_DIR_NAME
            / f"{song_id}.mid"
        ).is_file()
    assert int(chunks["n_songs"].sum()) == 2


def test_chunk_dataset_repack_rereads_flat_source(tmp_path: Path):
    source = tmp_path / "SPDMX_dev"
    dest = tmp_path / "SPDMX"
    source.mkdir()
    _write_flat_fixture(source)
    chunk_dataset(dataset_dir=source, package_dir=dest, target_bytes=50, seed=0)
    packaged, _chunks, assignment = chunk_dataset(
        dataset_dir=source,
        package_dir=dest,
        target_bytes=1,
        seed=1,
    )
    assert set(assignment) == {"0/1/QmA", "0/2/QmB"}
    assert (source / SPDMX_AUDIO_DIR_NAME / "0/1/QmA" / "0.flac").is_file()
    for song_id, chunk_id in assignment.items():
        assert (
            dest
            / chunk_dir_name(chunk_id)
            / SPDMX_AUDIO_DIR_NAME
            / song_id
            / "0.flac"
        ).is_file()
    assert "chunk" in packaged.columns


def test_stage_zenodo_files(tmp_path: Path):
    source = tmp_path / "SPDMX_dev"
    release = tmp_path / "SPDMX"
    stage = tmp_path / "zenodo"
    source.mkdir()
    _write_flat_fixture(source)
    chunk_dataset(dataset_dir=source, package_dir=release, target_bytes=50, seed=0)

    manifest = stage_zenodo_files(
        dataset_dir=release,
        stage_dir=stage,
        compression="store",
    )
    assert (stage / f"{SPDMX_FILE_NAME}.csv").is_file()
    assert (stage / CHUNKS_FILE_NAME).is_file()
    assert (stage / "LICENSE").is_file()
    assert (stage / "README.md").is_file()
    assert (stage / SHA256SUMS_FILE_NAME).is_file()

    zips = sorted(stage.glob("chunk_*.zip"))
    assert zips
    assert (manifest["archive"].astype(str) != "").all()
    assert (manifest["sha256"].astype(str) != "").all()

    with zipfile.ZipFile(zips[0]) as zf:
        names = zf.namelist()
    assert any(n.startswith("chunk_") and "/audio/" in n for n in names)
    assert any(n.startswith("chunk_") and "/mid/" in n for n in names)

    sums = (stage / SHA256SUMS_FILE_NAME).read_text(encoding="utf-8")
    for path in zips:
        assert path.name in sums


def test_stage_zenodo_subset_chunks(tmp_path: Path):
    source = tmp_path / "SPDMX_dev"
    release = tmp_path / "SPDMX"
    stage = tmp_path / "zenodo"
    source.mkdir()
    _write_flat_fixture(source)
    _, chunks, assignment = chunk_dataset(
        dataset_dir=source,
        package_dir=release,
        target_bytes=1,
        seed=0,
    )
    assert len(chunks) >= 2
    only = sorted(set(assignment.values()), key=int)[0]
    manifest = stage_zenodo_files(
        dataset_dir=release,
        stage_dir=stage,
        chunk_ids={only},
    )
    assert list(manifest["chunk"]) == [only]
    assert list(stage.glob("chunk_*.zip")) == [stage / f"chunk_{only}.zip"]


def test_build_spdmx_cli_defaults_to_release_dir(tmp_path: Path, capsys):
    from synthesis.build_spdmx import main as build_main

    out = tmp_path / "out"
    source = out / "SPDMX_dev"
    source.mkdir(parents=True)
    _write_flat_fixture(source)
    code = build_main(["-o", str(out), "--target-bytes", "50"])
    assert code == 0
    release = out / "SPDMX"
    assert (release / CHUNKS_FILE_NAME).is_file()
    assert (source / SPDMX_AUDIO_DIR_NAME / "0/1/QmA" / "0.flac").is_file()
    captured = capsys.readouterr()
    assert "left untouched" in captured.out


def test_build_spdmx_cli_dry_run(tmp_path: Path, capsys):
    from synthesis.build_spdmx import main as build_main

    out = tmp_path / "out"
    source = out / "SPDMX_dev"
    source.mkdir(parents=True)
    _write_flat_fixture(source)
    code = build_main(["-o", str(out), "--dry-run", "--target-bytes", "50"])
    assert code == 0
    captured = capsys.readouterr()
    assert "dry-run" in captured.out
    assert not (out / "SPDMX" / CHUNKS_FILE_NAME).exists()


def test_distribute_cli(tmp_path: Path):
    from synthesis.distribute_spdmx import main as dist_main

    source = tmp_path / "SPDMX_dev"
    release = tmp_path / "SPDMX"
    stage = tmp_path / "stage"
    source.mkdir()
    _write_flat_fixture(source)
    chunk_dataset(dataset_dir=source, package_dir=release, target_bytes=50, seed=0)
    code = dist_main(
        [
            "--dataset-dir",
            str(release),
            "--stage-dir",
            str(stage),
        ]
    )
    assert code == 0
    assert (stage / SHA256SUMS_FILE_NAME).is_file()
