"""Tests for post-render chunk assignment and packaging."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd
import pytest

from shared.config import (
    SPDMX_AUDIO_DIR_NAME,
    SPDMX_FILE_NAME,
    SPDMX_MID_DIR_NAME,
    SPDMX_SONGS_FILE_NAME,
)
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
    rewrite_songs_table_for_chunks,
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


def test_assign_songs_to_chunks_balanced_and_seeded():
    sizes = {
        "a/1/QmA": 10,
        "b/1/QmB": 10,
        "c/1/QmC": 10,
        "d/1/QmD": 5,
        "e/1/QmE": 5,
        "f/1/QmF": 8,
    }
    first = assign_songs_to_chunks(sizes, num_chunks=3, seed=1)
    second = assign_songs_to_chunks(sizes, num_chunks=3, seed=1)
    assert first == second
    assert set(first) == set(sizes)
    assert set(first.values()) == {"0", "1", "2"}
    loads = [
        sum(sizes[s] for s, c in first.items() if c == chunk_id)
        for chunk_id in ("0", "1", "2")
    ]
    assert max(loads) - min(loads) <= max(sizes.values())


def test_assign_songs_to_chunks_caps_at_n_songs():
    sizes = {"a": 1, "b": 2}
    assignment = assign_songs_to_chunks(sizes, num_chunks=64, seed=0)
    assert len(set(assignment.values())) == 2


def test_rewrite_track_map_for_chunks_updates_paths():
    table = pd.DataFrame(
        [
            {
                "song_id": "0/1/QmA",
                "path": "./audio/0/1/QmA",
                "mid": "./mid/0/1/QmA.mid",
                "mix": "./mix/0/1/QmA.flac",
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
                "mix": "./mix/0/1/QmA.flac",
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
    assert packaged.iloc[0]["mix"] == f"./chunk_0/0/1/QmA/mix.flac"
    assert packaged.iloc[0]["path"] == "./chunk_0/0/1/QmA"
    assert packaged.iloc[0]["mid"] == "./chunk_0/0/1/QmA/mix.mid"


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
    import numpy as np
    import soundfile as sf

    rows = []
    song_rows = []
    for song_id, tracks in (("0/1/QmA", 2), ("0/2/QmB", 1)):
        audio = root / SPDMX_AUDIO_DIR_NAME / song_id
        audio.mkdir(parents=True)
        for track in range(tracks):
            sf.write(
                str(audio / f"{track}.flac"),
                np.zeros(1000, dtype=np.float32),
                44100,
                format="FLAC",
            )
            rows.append(
                {
                    "song_id": song_id,
                    "path": f"./audio/{song_id}",
                    "mid": f"./mid/{song_id}.mid",
                    "mix": f"./mix/{song_id}.flac",
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
        mix = root / "mix" / f"{song_id}.flac"
        mix.parent.mkdir(parents=True, exist_ok=True)
        # Distinct lengths so song_length is a real mix-header value.
        n = 2000 if song_id.endswith("A") else 1000
        sf.write(str(mix), np.zeros(n, dtype=np.float32), 44100, format="FLAC")
        song_rows.append(
            {
                "song_id": song_id,
                "path": f"./audio/{song_id}",
                "mid": f"./mid/{song_id}.mid",
                "mix": f"./mix/{song_id}.flac",
                "n_tracks": tracks,
                "song_length": n / 44100.0,
                "subset:all": True,
                "subset:bdgp": False,
            }
        )
    pd.DataFrame(rows).to_csv(root / f"{SPDMX_FILE_NAME}.csv", index=False)
    pd.DataFrame(song_rows).to_csv(root / SPDMX_SONGS_FILE_NAME, index=False)
    write_spdmx_release_docs(root)


def test_rewrite_songs_table_for_chunks_keeps_song_length():
    songs = pd.DataFrame(
        [
            {
                "song_id": "0/1/QmA",
                "path": "./audio/0/1/QmA",
                "mid": "./mid/0/1/QmA.mid",
                "mix": "./mix/0/1/QmA.flac",
                "n_tracks": 2,
                "song_length": 12.5,
                "subset:all": True,
                "subset:bdgp": False,
            }
        ]
    )
    packaged = rewrite_songs_table_for_chunks(songs, {"0/1/QmA": "3"})
    assert packaged.iloc[0]["chunk"] == "3"
    assert packaged.iloc[0]["path"] == "./chunk_3/0/1/QmA"
    assert packaged.iloc[0]["mix"] == "./chunk_3/0/1/QmA/mix.flac"
    assert float(packaged.iloc[0]["song_length"]) == 12.5
    assert bool(packaged.iloc[0]["subset:multitrack"])


def test_rewrite_songs_table_derives_multitrack_subset():
    songs = pd.DataFrame(
        [
            {"song_id": "a", "n_tracks": 1, "path": "./audio/a", "mix": "./mix/a.flac"},
            {"song_id": "b", "n_tracks": 3, "path": "./audio/b", "mix": "./mix/b.flac"},
        ]
    )
    packaged = rewrite_songs_table_for_chunks(songs, {"a": "0", "b": "1"})
    assert not bool(packaged.loc[packaged["song_id"] == "a", "subset:multitrack"].iloc[0])
    assert bool(packaged.loc[packaged["song_id"] == "b", "subset:multitrack"].iloc[0])


def test_song_media_bytes_can_omit_mix(tmp_path: Path):
    from synthesis.chunking import song_media_bytes

    song_id = "0/1/QmA"
    audio = tmp_path / "audio" / song_id
    audio.mkdir(parents=True)
    (audio / "0.flac").write_bytes(b"a" * 100)
    mid = tmp_path / "mid" / f"{song_id}.mid"
    mid.parent.mkdir(parents=True)
    mid.write_bytes(b"m" * 20)
    mix = tmp_path / "mix" / f"{song_id}.flac"
    mix.parent.mkdir(parents=True)
    mix.write_bytes(b"x" * 50)
    with_mix = song_media_bytes(
        song_id, audio_root=tmp_path / "audio", mid_root=tmp_path / "mid",
        mix_root=tmp_path / "mix", include_mix=True,
    )
    without = song_media_bytes(
        song_id, audio_root=tmp_path / "audio", mid_root=tmp_path / "mid",
        mix_root=tmp_path / "mix", include_mix=False,
    )
    assert with_mix == without + 50
    assert without == 120


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
        chunk_dataset(dataset_dir=source, package_dir=source, num_chunks=2)


def test_chunk_dataset_builds_separate_release_tree(tmp_path: Path):
    source = tmp_path / "SPDMX_dev"
    dest = tmp_path / "SPDMX"
    source.mkdir()
    _write_flat_fixture(source)

    packaged, chunks, assignment = chunk_dataset(
        dataset_dir=source,
        package_dir=dest,
        num_chunks=2,
        seed=0,
    )
    assert set(assignment) == {"0/1/QmA", "0/2/QmB"}
    assert "chunk" in packaged.columns
    assert (dest / f"{SPDMX_FILE_NAME}.csv").is_file()
    assert (dest / CHUNKS_FILE_NAME).is_file()
    assert (dest / "LICENSE").is_file()
    songs = pd.read_csv(dest / "songs.csv")
    assert set(songs["song_id"]) == {"0/1/QmA", "0/2/QmB"}
    assert "subset:all" in songs.columns and "subset:bdgp" in songs.columns
    assert "subset:multitrack" in songs.columns
    assert "song_length" in songs.columns
    assert bool(songs["subset:all"].all())
    assert songs["song_length"].notna().all()
    assert (songs["song_length"] > 0).all()
    by_id = songs.set_index("song_id")
    assert bool(by_id.loc["0/1/QmA", "subset:multitrack"])
    assert not bool(by_id.loc["0/2/QmB", "subset:multitrack"])

    # Flat production tree untouched.
    assert (source / SPDMX_AUDIO_DIR_NAME / "0/1/QmA" / "0.flac").is_file()
    assert (source / SPDMX_MID_DIR_NAME / "0/1/QmA.mid").is_file()
    assert "chunk" not in pd.read_csv(source / f"{SPDMX_FILE_NAME}.csv").columns
    assert (source / SPDMX_SONGS_FILE_NAME).is_file()
    # Packaged songs keep precomputed song_length; paths are remapped.
    assert songs.iloc[0]["mix"].endswith("/mix.flac")
    assert str(songs.iloc[0]["path"]).startswith("./chunk_")

    for song_id, chunk_id in assignment.items():
        song_dir = dest / chunk_dir_name(chunk_id) / song_id
        assert (song_dir / "0.flac").is_file()
        assert (song_dir / "mix.mid").is_file()
        n_tracks = 2 if song_id.endswith("QmA") else 1
        if n_tracks >= 2:
            assert (song_dir / "mix.flac").is_file()
        else:
            assert not (song_dir / "mix.flac").exists()
        assert not (dest / chunk_dir_name(chunk_id) / SPDMX_AUDIO_DIR_NAME).exists()
        assert not (dest / chunk_dir_name(chunk_id) / SPDMX_MID_DIR_NAME).exists()
    assert (dest / "link_single_track_mixes.sh").is_file()
    assert int(chunks["n_songs"].sum()) == 2
    assert packaged.iloc[0]["path"].startswith("./chunk_")
    assert packaged.iloc[0]["path"].endswith("QmA") or packaged.iloc[0]["path"].endswith("QmB")
    assert packaged.iloc[0]["mid"].endswith("/mix.mid")
    assert packaged.iloc[0]["mix"].endswith("/mix.flac")


def test_chunk_dataset_removes_stale_single_track_mix(tmp_path: Path):
    source = tmp_path / "SPDMX_dev"
    dest = tmp_path / "SPDMX"
    source.mkdir()
    _write_flat_fixture(source)
    _, _, assignment = chunk_dataset(
        dataset_dir=source, package_dir=dest, num_chunks=2, seed=0,
    )
    single_dir = dest / chunk_dir_name(assignment["0/2/QmB"]) / "0/2/QmB"
    (single_dir / "mix.flac").write_bytes(b"stale")
    chunk_dataset(dataset_dir=source, package_dir=dest, num_chunks=2, seed=0)
    assert not (single_dir / "mix.flac").exists()
    multi_dir = dest / chunk_dir_name(assignment["0/1/QmA"]) / "0/1/QmA"
    assert (multi_dir / "mix.flac").is_file()


def test_chunk_dataset_repack_rereads_flat_source(tmp_path: Path):
    source = tmp_path / "SPDMX_dev"
    dest = tmp_path / "SPDMX"
    source.mkdir()
    _write_flat_fixture(source)
    chunk_dataset(dataset_dir=source, package_dir=dest, num_chunks=2, seed=0)
    packaged, _chunks, assignment = chunk_dataset(
        dataset_dir=source,
        package_dir=dest,
        num_chunks=2,
        seed=1,
    )
    assert set(assignment) == {"0/1/QmA", "0/2/QmB"}
    assert (source / SPDMX_AUDIO_DIR_NAME / "0/1/QmA" / "0.flac").is_file()
    for song_id, chunk_id in assignment.items():
        assert (dest / chunk_dir_name(chunk_id) / song_id / "0.flac").is_file()
        mix_path = dest / chunk_dir_name(chunk_id) / song_id / "mix.flac"
        if song_id.endswith("QmA"):
            assert mix_path.is_file()
        else:
            assert not mix_path.exists()
    assert "chunk" in packaged.columns


def test_stage_zenodo_files(tmp_path: Path):
    source = tmp_path / "SPDMX_dev"
    release = tmp_path / "SPDMX"
    stage = tmp_path / "zenodo"
    source.mkdir()
    _write_flat_fixture(source)
    chunk_dataset(dataset_dir=source, package_dir=release, num_chunks=2, seed=0)

    manifest = stage_zenodo_files(
        dataset_dir=release,
        stage_dir=stage,
        compression="store",
    )
    assert (stage / f"{SPDMX_FILE_NAME}.csv").is_file()
    assert (stage / CHUNKS_FILE_NAME).is_file()
    assert (stage / "LICENSE").is_file()
    assert (stage / "README.md").is_file()
    assert (stage / "link_single_track_mixes.sh").is_file()
    assert (stage / "songs.csv").is_file()
    assert (stage / SHA256SUMS_FILE_NAME).is_file()

    zips = sorted(stage.glob("chunk_*.zip"))
    assert zips
    assert (manifest["archive"].astype(str) != "").all()
    assert (manifest["sha256"].astype(str) != "").all()

    all_names: list[str] = []
    for zpath in zips:
        with zipfile.ZipFile(zpath) as zf:
            all_names.extend(zf.namelist())
    assert any(n.startswith("chunk_") and "/mix.mid" in n for n in all_names)
    assert any(n.startswith("chunk_") and n.endswith("/0.flac") for n in all_names)
    # Multitrack QmA ships mix.flac; single-track QmB does not.
    assert any(n.endswith("QmA/mix.flac") for n in all_names)
    assert not any(n.endswith("QmB/mix.flac") for n in all_names)
    assert not any("/audio/" in n for n in all_names)

    sums = (stage / SHA256SUMS_FILE_NAME).read_text(encoding="utf-8")
    for path in zips:
        assert path.name in sums
    assert "link_single_track_mixes.sh" in sums


def test_stage_zenodo_subset_chunks(tmp_path: Path):
    source = tmp_path / "SPDMX_dev"
    release = tmp_path / "SPDMX"
    stage = tmp_path / "zenodo"
    source.mkdir()
    _write_flat_fixture(source)
    _, chunks, assignment = chunk_dataset(
        dataset_dir=source,
        package_dir=release,
        num_chunks=2,
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
    code = build_main(["-o", str(out), "--num-chunks", "2"])
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
    code = build_main(["-o", str(out), "--dry-run", "--num-chunks", "2"])
    assert code == 0
    captured = capsys.readouterr()
    assert "dry-run" in captured.out
    assert not (out / "SPDMX" / CHUNKS_FILE_NAME).exists()


def test_link_single_track_mixes_script(tmp_path: Path):
    source = tmp_path / "SPDMX_dev"
    release = tmp_path / "SPDMX"
    source.mkdir()
    _write_flat_fixture(source)
    _, _, assignment = chunk_dataset(
        dataset_dir=source, package_dir=release, num_chunks=2, seed=0,
    )
    script = release / "link_single_track_mixes.sh"
    assert script.is_file()
    import subprocess

    result = subprocess.run(
        ["bash", str(script), str(release)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "1 single-track" in result.stdout
    single = release / chunk_dir_name(assignment["0/2/QmB"]) / "0/2/QmB" / "mix.flac"
    assert single.is_symlink()
    assert single.readlink().name == "0.flac"
    multi = release / chunk_dir_name(assignment["0/1/QmA"]) / "0/1/QmA" / "mix.flac"
    assert multi.is_file() and not multi.is_symlink()


def test_distribute_cli(tmp_path: Path):
    from synthesis.distribute_spdmx import main as dist_main

    source = tmp_path / "SPDMX_dev"
    release = tmp_path / "SPDMX"
    stage = tmp_path / "stage"
    source.mkdir()
    _write_flat_fixture(source)
    chunk_dataset(dataset_dir=source, package_dir=release, num_chunks=2, seed=0)
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
    assert (stage / "link_single_track_mixes.sh").is_file()
    assert (stage / "songs.csv").is_file()
