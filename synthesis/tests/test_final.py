"""CLI tests for hybrid final synthesis."""

from pathlib import Path

import pandas as pd
import pytest

from shared.config import DATA_DIR_NAME, STEMS_FILE_NAME
from synthesis.final import (
    FINAL_CONDITION,
    expected_song_count,
    hybrid_dirs,
    parse_args,
    pass_sequence,
    raw_upstream_command,
)
from synthesis.paths import (
    MIDI_INDEX_FILE_NAME,
    ablation_raw_dir,
    production_tables_dir,
    spdmx_dataset_dir,
)
from synthesis.recipe import (
    DEFAULT_RECIPE_PATH,
    STEM_RECIPE_FILE_NAME,
    CategoryRecipe,
    CategorySpec,
)
from synthesis.synthesize import attach_corrected_midi, run_layout_pass


def test_parse_args_accepts_merge():
    args = parse_args(["--only-pass", "merge"])
    assert args.only_pass == "merge"


def test_parse_args_requires_only_pass():
    with pytest.raises(SystemExit):
        parse_args([])


def test_parse_args_layout_defaults_full_flac():
    args = parse_args(["--only-pass", "layout"])
    assert args.full is True
    assert args.flac is True
    assert args.recipe == str(DEFAULT_RECIPE_PATH)
    assert args.only_pass == "layout"
    assert args.yes is False


def test_parse_args_mp3_rejected_flac_always():
    with pytest.raises(SystemExit):
        parse_args(["--only-pass", "fluidsynth", "--mp3"])
    args = parse_args(["--ablation-sample", "--only-pass", "fluidsynth", "-j", "4", "-y"])
    assert args.full is False
    assert args.flac is True
    assert args.only_pass == "fluidsynth"
    assert args.jobs == 4
    assert args.yes is True


def test_hybrid_dirs_write_one_tree():
    full = parse_args(["-o", "/tmp/spdmx", "--only-pass", "layout"])
    sample = parse_args(["-o", "/tmp/spdmx", "--ablation-sample", "--only-pass", "layout"])
    assert hybrid_dirs(full) == (
        production_tables_dir("/tmp/spdmx"),
        spdmx_dataset_dir("/tmp/spdmx"),
    )
    dest = ablation_raw_dir("/tmp/spdmx", FINAL_CONDITION)
    assert hybrid_dirs(sample) == (dest, dest)


def test_pass_sequence_starts_with_layout():
    no_realify = CategoryRecipe(
        specs={"piano": CategorySpec("basic", False, "basic", "basic")},
    )
    with_realify = CategoryRecipe(
        specs={"piano": CategorySpec("basic", True, "basic", "basic_realify")},
    )
    with_ddsp = CategoryRecipe(
        specs={"strings": CategorySpec("midi-ddsp", False, "basic", "ddsp_basic")},
    )
    with_ddsp_realify = CategoryRecipe(
        specs={"strings": CategorySpec("midi-ddsp", True, "basic", "ddsp_basic_realify")},
    )
    assert pass_sequence(no_realify) == ("layout", "fluidsynth", "mix")
    assert pass_sequence(with_ddsp) == (
        "layout", "fluidsynth", "midi_ddsp", "mix",
    )
    assert pass_sequence(with_realify) == ("layout", "fluidsynth", "realify", "mix")
    assert pass_sequence(with_ddsp_realify) == (
        "layout", "fluidsynth", "midi_ddsp", "realify", "mix",
    )
    with_piano_ddsp = CategoryRecipe(
        specs={
            "piano": CategorySpec("midi-ddsp", False, "basic", "ddsp_basic"),
            "strings": CategorySpec("midi-ddsp", False, "basic", "ddsp_basic"),
        },
    )
    assert pass_sequence(with_piano_ddsp) == (
        "layout", "fluidsynth", "ddsp_piano", "midi_ddsp", "mix",
    )


def test_layout_pass_creates_song_dirs(tmp_path: Path):
    pdmx_root = tmp_path / "PDMX"
    pdmx_root.mkdir()
    csv_path = pdmx_root / "PDMX.csv"
    pd.DataFrame({
        "path": ["./data/1/11/QmTestSong.json"],
        "mid": ["./mid/1/11/QmTestSong.mid"],
        "subset:all_valid": [True],
        "n_tracks": [2],
    }).to_csv(csv_path, index=False)

    out = tmp_path / "SPDMX"
    args = parse_args([
        "--only-pass", "layout",
        "-o", str(out),
        "-df", str(csv_path),
        "--no-register",
        "-j", "2",
    ])
    args.recipe = CategoryRecipe(
        specs={"piano": CategorySpec("basic", False, "basic", "basic")},
    )
    dest = spdmx_dataset_dir(str(out))
    tables = production_tables_dir(str(out))
    dataset = run_layout_pass(args, tables, media_dir=dest)
    song_dir = Path(dataset.iloc[0]["path_output"])
    assert song_dir.is_dir()
    assert song_dir == Path(dest) / "audio" / "1" / "11" / "QmTestSong"
    assert (Path(dest) / "mid" / "1" / "11").is_dir()
    assert not (Path(dest) / f"{DATA_DIR_NAME}.csv").is_file()
    assert (Path(tables) / f"{DATA_DIR_NAME}.csv").is_file()
    assert (Path(tables) / f"{STEMS_FILE_NAME}.csv").is_file()
    assert (Path(tables) / STEM_RECIPE_FILE_NAME).is_file()
    assert (Path(dest) / "LICENSE").is_file()
    assert (Path(dest) / "README.md").is_file()


def test_layout_pass_restricts_to_spdmx_csv(tmp_path: Path):
    pdmx_root = tmp_path / "PDMX"
    pdmx_root.mkdir()
    csv_path = pdmx_root / "PDMX.csv"
    pd.DataFrame({
        "path": ["./data/1/11/QmKeep.json", "./data/1/11/QmDrop.json"],
        "mid": ["./mid/1/11/QmKeep.mid", "./mid/1/11/QmDrop.mid"],
        "subset:all_valid": [True, True],
        "n_tracks": [1, 1],
    }).to_csv(csv_path, index=False)

    out = tmp_path / "out"
    dest = Path(spdmx_dataset_dir(str(out)))
    dest.mkdir(parents=True)
    pd.DataFrame({
        "song_id": ["1/11/QmKeep"],
        "path": ["./audio/1/11/QmKeep"],
        "mid": ["./mid/1/11/QmKeep.mid"],
        "track": [0],
        "original_track": [0],
        "program": [0],
        "is_drum": [False],
        "name": ["Piano"],
    }).to_csv(dest / "SPDMX.csv", index=False)

    args = parse_args([
        "--only-pass", "layout",
        "-o", str(out),
        "-df", str(csv_path),
        "--no-register",
        "-j", "1",
    ])
    args.recipe = CategoryRecipe(
        specs={"piano": CategorySpec("basic", False, "basic", "basic")},
    )
    dataset = run_layout_pass(args, production_tables_dir(str(out)), media_dir=str(dest))
    assert len(dataset) == 1
    assert Path(dataset.iloc[0]["path_output"]).name == "QmKeep"
    assert (Path(dest) / "audio" / "1" / "11" / "QmKeep").is_dir()
    assert not (Path(dest) / "audio" / "1" / "11" / "QmDrop").is_dir()
    index_path = Path(production_tables_dir(str(out))) / MIDI_INDEX_FILE_NAME
    assert index_path.is_file()
    index = pd.read_csv(index_path)
    assert list(index["song_id"]) == ["1/11/QmKeep"]
    assert int(index.iloc[0]["n_tracks"]) == 1


def test_attach_corrected_midi_uses_index_without_stat(tmp_path: Path):
    dest = tmp_path / "SPDMX"
    mid_dir = dest / "mid"
    mid_dir.mkdir(parents=True)
    pd.DataFrame({
        "song_id": ["1/11/QmA", "1/11/QmA", "2/22/QmB"],
        "path": ["./audio/1/11/QmA", "./audio/1/11/QmA", "./audio/2/22/QmB"],
        "mid": ["./mid/1/11/QmA.mid", "./mid/1/11/QmA.mid", "./mid/2/22/QmB.mid"],
        "track": [0, 1, 0],
        "original_track": [0, 1, 0],
        "program": [0, 0, 0],
        "is_drum": [False, False, False],
        "name": ["A", "B", "C"],
    }).to_csv(dest / "SPDMX.csv", index=False)
    tables = tmp_path / "dev" / "final"
    tables.mkdir(parents=True)
    args = parse_args(["--only-pass", "fluidsynth", "-o", str(tmp_path), "--no-register"])
    dataset = pd.DataFrame({
        "mid_pdmx": [
            "/pdmx/mid/1/11/QmA.mid",
            "/pdmx/mid/2/22/QmB.mid",
        ],
        "path_output": ["/audio/a", "/audio/b"],
    })
    out = attach_corrected_midi(dataset, args, str(tables))
    assert int(out.iloc[0]["n_tracks"]) == 2
    assert int(out.iloc[1]["n_tracks"]) == 1
    assert out.iloc[0]["mid"].endswith("/mid/1/11/QmA.mid")
    assert (tables / MIDI_INDEX_FILE_NAME).is_file()
    assert int(out.iloc[0]["n_fluidsynth"]) == 2
    assert int(out.iloc[1]["n_fluidsynth"]) == 1


def test_work_for_pass_counts_remaining_renders():
    from synthesis.synthesize import _work_for_pass

    df = pd.DataFrame({
        "path_output": ["/a", "/b"],
        "n_fluidsynth": [3, 0],
        "n_ddsp_piano": [0, 1],
        "n_midi_ddsp": [0, 4],
    })
    fs, fs_n = _work_for_pass(df, [0, 1], "fluidsynth")
    assert fs == [0, 1] and fs_n == 3
    piano, pn = _work_for_pass(df, [0, 1], "ddsp_piano")
    assert piano == [1] and pn == 1
    midi, mn = _work_for_pass(df, [0, 1], "midi_ddsp")
    assert midi == [1] and mn == 4
    recipe_index = {
        ("/b", 0): {"backend": "midi_ddsp"},
        ("/b", 1): {"backend": "midi_ddsp"},
    }
    _, remaining = _work_for_pass(
        df, [0, 1], "midi_ddsp", stem_recipe_index=recipe_index,
    )
    assert remaining == 2


def test_song_index_needs_work_hybrid_midi_ddsp():
    from synthesis.recipe import CategoryRecipe, CategorySpec
    from synthesis.synthesize import _song_index_needs_work

    df = pd.DataFrame({
        "path_output": ["/a", "/b"],
        "n_tracks": [2, 2],
        "n_midi_ddsp": [0, 4],
    })
    recipe = CategoryRecipe(
        specs={"strings": CategorySpec("midi-ddsp", True, "basic", "ddsp_basic_realify")},
    )
    args = parse_args(["--only-pass", "midi_ddsp", "-o", "/tmp"])
    args.recipe = recipe
    args.ddsp_pass = "midi_ddsp"
    args.reset = False
    args.stem_recipe_index = {
        ("/b", 0): {"backend": "midi_ddsp"},
        ("/b", 1): {"backend": "midi_ddsp"},
        ("/b", 2): {"backend": "midi_ddsp"},
        ("/b", 3): {"backend": "midi_ddsp"},
    }
    assert not _song_index_needs_work(1, df, args, set(), "flac")
    args.stem_recipe_index = {( "/b", 0): {"backend": "midi_ddsp"}}
    assert _song_index_needs_work(1, df, args, set(), "flac")


def test_reload_progress_from_disk_reads_pass_recipe(tmp_path: Path):
    from synthesis.recipe import STEM_RECIPE_COLUMNS
    from synthesis.synthesize import _reload_progress_from_disk

    tables = tmp_path / "dev" / "final"
    tables.mkdir(parents=True)
    recipe_csv = tables / "stem_recipe.midi_ddsp.csv"
    pd.DataFrame({
        "path": ["/song"],
        "track": [0],
        "category": ["strings"],
        "method": ["midi-ddsp"],
        "fallback": ["basic"],
        "realify": [True],
        "backend": ["midi_ddsp"],
    }).to_csv(recipe_csv, index=False)
    args = parse_args(["--only-pass", "midi_ddsp", "-o", str(tmp_path)])
    args.ddsp_pass = "midi_ddsp"
    args.reset = False
    args.stem_recipe_index = {}
    out = _reload_progress_from_disk(
        args,
        output_filepath=str(tables / "data.csv"),
        routing_output_filepath=None,
        recipe_output_filepath=str(recipe_csv),
        completed_paths=set(),
    )
    assert out == set()
    assert args.stem_recipe_index[("/song", 0)]["backend"] == "midi_ddsp"


def test_raw_upstream_command_includes_ddsp_when_needed():
    fluidsynth_only = CategoryRecipe(
        specs={"piano": CategorySpec("basic", True, "basic", "basic_realify")},
    )
    with_ddsp = CategoryRecipe(
        specs={"strings": CategorySpec("midi-ddsp", True, "basic", "ddsp_basic_realify")},
    )
    assert "ddsp" not in raw_upstream_command(fluidsynth_only)
    assert "fluidsynth" in raw_upstream_command(fluidsynth_only)
    cmd = raw_upstream_command(with_ddsp)
    assert "--only-pass fluidsynth" in cmd
    assert "--only-pass ddsp_piano" not in cmd
    assert "--only-pass midi_ddsp" in cmd


def test_expected_song_count_from_spdmx_csv(tmp_path: Path):
    dest = tmp_path / "SPDMX"
    dest.mkdir()
    pd.DataFrame({
        "song_id": ["a/b/QmOne", "a/b/QmOne", "a/b/QmTwo"],
        "track": [0, 1, 0],
    }).to_csv(dest / "SPDMX.csv", index=False)
    args = parse_args(["--only-pass", "realify", "-o", str(tmp_path)])
    assert expected_song_count(args, str(dest)) == 2
