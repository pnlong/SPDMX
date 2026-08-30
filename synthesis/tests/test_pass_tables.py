"""Tests for per-pass CSV shards and merge."""

from pathlib import Path

import pandas as pd

from shared.config import DATA_DIR_NAME, STEMS_FILE_NAME, STEMS_TABLE_COLUMNS
from synthesis.pass_tables import merge_pass_tables, pass_recipe_csv, pass_stems_csv
from synthesis.paths import MIDI_INDEX_FILE_NAME
from synthesis.recipe import STEM_RECIPE_COLUMNS, STEM_RECIPE_FILE_NAME


def _stem(path: str, track: int, name: str) -> dict:
    return {
        "path": path, "track": track, "original_track": track, "program": track,
        "is_drum": False, "name": name, "has_lyrics": False,
        "max_velocity": 64, "velocity_scale": 0.5,
    }


def _recipe(path: str, track: int, backend: str) -> dict:
    return {
        "path": path, "track": track, "category": "piano", "ablation": "basic",
        "method": "basic", "fallback": "basic", "backend": backend, "realify": False,
    }


def test_merge_pass_tables_uses_shards_only(tmp_path: Path):
    tables = tmp_path / "final"
    tables.mkdir()
    song = "/out/SPDMX/raw/7/19/QmSong"
    pd.DataFrame({
        "song_id": ["7/19/QmSong"],
        "n_tracks": [2],
    }).to_csv(tables / MIDI_INDEX_FILE_NAME, index=False)

    pd.DataFrame([_stem(song, 0, "stale")]).to_csv(
        tables / f"{STEMS_FILE_NAME}.csv", index=False,
    )
    pd.DataFrame([_stem(song, 0, "fluidsynth"), _stem(song, 1, "strings")]).to_csv(
        pass_stems_csv(tables, "fluidsynth"), index=False,
    )
    pd.DataFrame([_recipe(song, 0, "fluidsynth")]).to_csv(
        pass_recipe_csv(tables, "fluidsynth"), index=False,
    )
    pd.DataFrame([_recipe(song, 1, "midi_ddsp")]).to_csv(
        pass_recipe_csv(tables, "midi_ddsp"), index=False,
    )

    shard = pass_stems_csv(tables, "fluidsynth")
    shard_before = shard.read_text()
    recipe_shard = pass_recipe_csv(tables, "fluidsynth")
    recipe_before = recipe_shard.read_text()

    counts = merge_pass_tables(tables)
    assert counts["stems"] == 2
    assert counts["songs"] == 1
    assert shard.read_text() == shard_before
    assert recipe_shard.read_text() == recipe_before
    assert pass_recipe_csv(tables, "midi_ddsp").is_file()
    stems = pd.read_csv(tables / f"{STEMS_FILE_NAME}.csv")
    assert stems[stems["track"] == 0].iloc[0]["name"] == "fluidsynth"
    recipes = pd.read_csv(tables / STEM_RECIPE_FILE_NAME)
    assert set(recipes["backend"]) == {"fluidsynth", "midi_ddsp"}
    songs = pd.read_csv(tables / f"{DATA_DIR_NAME}.csv")
    assert len(songs) == 1
    assert int(songs.iloc[0]["n_tracks"]) == 2


def test_drop_canonical_tables(tmp_path: Path):
    from synthesis.pass_tables import drop_canonical_tables

    tables = tmp_path / "final"
    tables.mkdir()
    (tables / f"{STEMS_FILE_NAME}.csv").write_text("path\n")
    (tables / STEM_RECIPE_FILE_NAME).write_text("path\n")
    (tables / f"{DATA_DIR_NAME}.csv").write_text("path\n")
    (tables / f"{STEMS_FILE_NAME}.fluidsynth.csv").write_text("keep\n")
    drop_canonical_tables(tables)
    assert not (tables / f"{STEMS_FILE_NAME}.csv").exists()
    assert not (tables / STEM_RECIPE_FILE_NAME).exists()
    assert (tables / f"{STEMS_FILE_NAME}.fluidsynth.csv").read_text() == "keep\n"


def test_merge_pass_tables_incomplete_song_stays_out_of_data_csv(tmp_path: Path):
    tables = tmp_path / "final"
    tables.mkdir()
    song = "/out/SPDMX/raw/a/b/QmPartial"
    pd.DataFrame({"song_id": ["a/b/QmPartial"], "n_tracks": [3]}).to_csv(
        tables / MIDI_INDEX_FILE_NAME, index=False,
    )
    pd.DataFrame([_stem(song, 0, "piano")]).to_csv(
        pass_stems_csv(tables, "fluidsynth"), index=False,
    )
    counts = merge_pass_tables(tables)
    assert counts["songs"] == 0
    songs = pd.read_csv(tables / f"{DATA_DIR_NAME}.csv")
    assert len(songs) == 0
    assert list(STEMS_TABLE_COLUMNS) == list(pd.read_csv(
        tables / f"{STEMS_FILE_NAME}.csv",
    ).columns)
    assert list(STEM_RECIPE_COLUMNS) == list(pd.read_csv(
        tables / STEM_RECIPE_FILE_NAME,
    ).columns)
