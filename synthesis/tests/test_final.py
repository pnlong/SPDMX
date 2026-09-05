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


def test_parse_args_resume_check_disk_default():
    default = parse_args(["--only-pass", "midi_ddsp"])
    assert default.resume_check_disk is True
    off = parse_args(["--only-pass", "midi_ddsp", "--no-resume-check-disk"])
    assert off.resume_check_disk is False
    on = parse_args(["--only-pass", "midi_ddsp", "--resume-check-disk"])
    assert on.resume_check_disk is True


def test_hybrid_raw_current_csv_only_skips_without_disk(tmp_path: Path):
    from synthesis.recipe import CategorySpec, TrackPlan
    from synthesis.synthesize import _hybrid_raw_current

    plan = TrackPlan(
        category="strings",
        method="midi-ddsp",
        realify=False,
        use_slakh=False,
        neural_ok=True,
        fallback="basic",
        ablation="ddsp_basic",
    )
    missing = tmp_path / "nope.flac"
    args = parse_args([
        "--only-pass", "midi_ddsp", "-o", str(tmp_path), "--no-resume-check-disk",
    ])
    args.reset = False
    args.stem_recipe_index = {
        ("/song", 0): {
            "method": "midi-ddsp",
            "fallback": "basic",
            "backend": "midi_ddsp",
        },
    }
    assert args.resume_check_disk is False
    assert _hybrid_raw_current(args, "/song", 0, missing, plan, "midi_ddsp")

    args.resume_check_disk = True
    assert not _hybrid_raw_current(args, "/song", 0, missing, plan, "midi_ddsp")


def test_verify_claimed_stems_on_disk_raises_when_missing(tmp_path: Path):
    from synthesis.synthesize import verify_claimed_stems_on_disk

    tables = tmp_path / "final"
    tables.mkdir()
    song = tmp_path / "audio" / "a" / "b" / "Qm"
    song.mkdir(parents=True)
    pd.DataFrame({
        "path": [str(song)],
        "track": [0],
        "original_track": [0],
        "program": [0],
        "is_drum": [False],
        "name": ["x"],
        "has_lyrics": [False],
        "max_velocity": [80],
        "velocity_scale": [1.0],
    }).to_csv(tables / "stems.fluidsynth.csv", index=False)
    with pytest.raises(RuntimeError, match="missing or invalid") as excinfo:
        verify_claimed_stems_on_disk(tables, "flac")
    assert str(song / "0.flac") in str(excinfo.value) or "0.flac" in str(excinfo.value)


def test_count_pass_remaining_and_verify_reports(tmp_path: Path):
    from synthesis.audio import save_stem
    from synthesis.pass_tables import pass_recipe_csv, pass_stems_csv
    from synthesis.paths import MIDI_INDEX_FILE_NAME
    from synthesis.recipe import CategoryRecipe, CategorySpec
    from synthesis.synthesize import count_pass_remaining, verify_claimed_stems_on_disk
    import torch

    tables = tmp_path / "final"
    tables.mkdir()
    song = tmp_path / "SPDMX" / "audio" / "1" / "11" / "QmA"
    song.mkdir(parents=True)
    path = str(song)
    pd.DataFrame({
        "song_id": ["1/11/QmA"],
        "mid": [str(tmp_path / "SPDMX" / "mid" / "1" / "11" / "QmA.mid")],
        "n_tracks": [2],
        "n_fluidsynth": [2],
        "n_ddsp_piano": [0],
        "n_midi_ddsp": [0],
    }).to_csv(tables / MIDI_INDEX_FILE_NAME, index=False)
    # One of two fluidsynth stems recorded → 1 remaining
    pd.DataFrame([{
        "path": path,
        "track": 0,
        "category": "piano",
        "ablation": "basic",
        "method": "basic",
        "fallback": "basic",
        "backend": "fluidsynth",
        "realify": False,
    }]).to_csv(pass_recipe_csv(tables, "fluidsynth"), index=False)
    recipe = CategoryRecipe(
        specs={"piano": CategorySpec("basic", False, "basic", "basic")},
    )
    rows = count_pass_remaining(tables, recipe=recipe)
    assert rows[0]["pass"] == "fluidsynth"
    assert rows[0]["remaining"] == 1
    assert rows[0]["examples"] == [{"song_id": "1/11/QmA", "remaining": 1}]
    with pytest.raises(RuntimeError, match=r"1/11/QmA") as excinfo:
        verify_claimed_stems_on_disk(tables, "flac", recipe=recipe)
    assert "stems left to do" in str(excinfo.value)

    # Finish recipe + write both stems on disk → verify passes
    pd.DataFrame([
        {
            "path": path, "track": t, "category": "piano", "ablation": "basic",
            "method": "basic", "fallback": "basic", "backend": "fluidsynth",
            "realify": False,
        }
        for t in (0, 1)
    ]).to_csv(pass_recipe_csv(tables, "fluidsynth"), index=False)
    save_stem(torch.zeros(1, 100), song, 0, "flac")
    save_stem(torch.zeros(1, 100), song, 1, "flac")
    pd.DataFrame({
        "path": [path, path],
        "track": [0, 1],
        "original_track": [0, 1],
        "program": [0, 0],
        "is_drum": [False, False],
        "name": ["a", "b"],
        "has_lyrics": [False, False],
        "max_velocity": [80, 80],
        "velocity_scale": [1.0, 1.0],
    }).to_csv(pass_stems_csv(tables, "fluidsynth"), index=False)
    verify_claimed_stems_on_disk(tables, "flac", recipe=recipe)
    assert count_pass_remaining(tables, recipe=recipe)[0]["remaining"] == 0


def test_raw_path_to_audio():
    from synthesis.paths import raw_path_to_audio

    assert raw_path_to_audio("./raw/1/11/Qm") == "./audio/1/11/Qm"
    assert (
        raw_path_to_audio("/deepfreeze/share/SPDMX/SPDMX/raw/1/11/Qm")
        == "/deepfreeze/share/SPDMX/SPDMX/audio/1/11/Qm"
    )


def test_hybrid_mix_writes_audio_leaves_raw(tmp_path: Path):
    """Final mix writes to SPDMX/audio and leaves SPDMX/raw untouched."""
    from shared.config import SAMPLE_RATE, SPDMX_FILE_NAME
    from synthesis.audio import load_stem
    from synthesis.final import run_summable_mix
    import numpy as np
    import soundfile as sf

    media = tmp_path / "SPDMX"
    tables = tmp_path / "dev" / "final"
    tables.mkdir(parents=True)
    song_rel = "1/11/QmMix"
    raw_song = media / "raw" / song_rel
    raw_song.mkdir(parents=True)
    sr = SAMPLE_RATE
    sf.write(str(raw_song / "0.flac"), np.full(sr, 0.9, np.float32), sr, format="FLAC")
    sf.write(str(raw_song / "1.flac"), np.full(sr, 0.9, np.float32), sr, format="FLAC")
    path = str(raw_song)
    pd.DataFrame({
        "path": [path, path],
        "track": [0, 1],
        "velocity_scale": [1.0, 1.0],
    }).to_csv(tables / "stems.csv", index=False)
    pd.DataFrame({"path": [path], "n_tracks": [2]}).to_csv(tables / "data.csv", index=False)
    pd.DataFrame({
        "song_id": [song_rel],
        "path": [f"./raw/{song_rel}"],
        "track": [0],
    }).to_csv(media / f"{SPDMX_FILE_NAME}.csv", index=False)

    class _Args:
        jobs = 1
        dataset_filepath = str(tmp_path / "PDMX.csv")
        output_dir = str(tmp_path)

    run_summable_mix(_Args(), str(tables), media_dir=str(media))

    audio_song = media / "audio" / song_rel
    assert (audio_song / "0.flac").is_file()
    assert (audio_song / "1.flac").is_file()
    # Raw still loud / untouched peak region
    assert load_stem(raw_song / "0.flac").abs().max().item() > 0.8
    # Mixable peak-limited when summed
    s0 = load_stem(audio_song / "0.flac")
    s1 = load_stem(audio_song / "1.flac")
    assert (s0 + s1).abs().max().item() <= 1.0 + 1e-4
    # Pipeline tables still point at raw/
    assert pd.read_csv(tables / "stems.csv").iloc[0]["path"] == path
    # Released SPDMX.csv points at audio/
    released = pd.read_csv(media / f"{SPDMX_FILE_NAME}.csv")
    assert released.iloc[0]["path"] == f"./audio/{song_rel}"


def test_parse_args_accepts_verify():
    args = parse_args(["--only-pass", "verify"])
    assert args.only_pass == "verify"


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
    assert pass_sequence(no_realify) == ("layout", "fluidsynth", "verify", "mix")
    assert pass_sequence(with_ddsp) == (
        "layout", "fluidsynth", "midi_ddsp", "verify", "mix",
    )
    assert pass_sequence(with_realify) == (
        "layout", "fluidsynth", "realify", "verify", "mix",
    )
    assert pass_sequence(with_ddsp_realify) == (
        "layout", "fluidsynth", "midi_ddsp", "realify", "verify", "mix",
    )
    with_piano_ddsp = CategoryRecipe(
        specs={
            "piano": CategorySpec("midi-ddsp", False, "basic", "ddsp_basic"),
            "strings": CategorySpec("midi-ddsp", False, "basic", "ddsp_basic"),
        },
    )
    assert pass_sequence(with_piano_ddsp) == (
        "layout", "fluidsynth", "ddsp_piano", "midi_ddsp", "verify", "mix",
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
    assert song_dir == Path(dest) / "raw" / "1" / "11" / "QmTestSong"
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
    assert (Path(dest) / "raw" / "1" / "11" / "QmKeep").is_dir()
    assert not (Path(dest) / "raw" / "1" / "11" / "QmDrop").is_dir()
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


def test_work_for_pass_counts_remaining_renders(tmp_path: Path):
    from synthesis.synthesize import _work_for_pass

    df = pd.DataFrame({
        "path_output": ["/a", "/b"],
        "n_fluidsynth": [3, 0],
        "n_ddsp_piano": [0, 1],
        "n_midi_ddsp": [0, 4],
    })
    # Fluidsynth visits pure-MIDI-DDSP songs to claim soundfont fallbacks,
    # but bar total only counts native SF tracks (not neural deferrals).
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
    # Fluidsynth resume: skip songs already covered by stem_recipe.fluidsynth.csv
    fs_done = {
        ("/a", 0): {"method": "basic", "backend": "fluidsynth"},
        ("/a", 1): {"method": "basic", "backend": "fluidsynth"},
        ("/a", 2): {"method": "basic", "backend": "fluidsynth"},
    }
    fs2, fs2_n = _work_for_pass(
        df, [0, 1], "fluidsynth", stem_recipe_index=fs_done,
    )
    # /a native done; /b still has 4 unclaimed layout-MIDI-DDSP tracks (bar=0 native)
    assert fs2 == [1] and fs2_n == 0

    # MIDI-DDSP credits Fluidsynth method=midi-ddsp polyphony fallbacks.
    tables = tmp_path / "final"
    tables.mkdir()
    pd.DataFrame({
        "path": ["/b", "/b"],
        "track": [2, 3],
        "category": ["strings", "strings"],
        "ablation": ["ddsp_basic", "ddsp_basic"],
        "method": ["midi-ddsp", "midi-ddsp"],
        "fallback": ["basic", "basic"],
        "backend": ["fluidsynth", "fluidsynth"],
        "realify": [False, False],
        "reason": ["soundfont_polyphonic", "soundfont_polyphonic"],
    }).to_csv(tables / "stem_recipe.fluidsynth.csv", index=False)
    _, rem_fb = _work_for_pass(
        df, [0, 1], "midi_ddsp",
        stem_recipe_index=recipe_index,
        tables_dir=tables,
    )
    assert rem_fb == 0

    # Fluidsynth picks up leftover layout-MIDI-DDSP soundfont fallbacks (orphan case).
    fs_partial = {
        ("/a", 0): {"method": "basic", "backend": "fluidsynth"},
        ("/a", 1): {"method": "basic", "backend": "fluidsynth"},
        ("/a", 2): {"method": "basic", "backend": "fluidsynth"},
        ("/b", 1): {"method": "midi-ddsp", "backend": "fluidsynth"},
        ("/b", 2): {"method": "midi-ddsp", "backend": "fluidsynth"},
        ("/b", 3): {"method": "midi-ddsp", "backend": "fluidsynth"},
    }
    fs3, fs3_n = _work_for_pass(
        df, [0, 1], "fluidsynth",
        stem_recipe_index=fs_partial,
        tables_dir=tables,
    )
    # 1 neural track unclaimed → song kept, but bar total=0 (native only)
    assert fs3 == [1] and fs3_n == 0

    # Pending (deferred-to-neural) rows clear Fluidsynth work without counting as SF fallbacks.
    fs_pending = {
        ("/a", 0): {"method": "basic", "backend": "fluidsynth"},
        ("/a", 1): {"method": "basic", "backend": "fluidsynth"},
        ("/a", 2): {"method": "basic", "backend": "fluidsynth"},
        ("/b", 0): {"method": "midi-ddsp", "backend": "pending_midi_ddsp"},
        ("/b", 1): {"method": "midi-ddsp", "backend": "pending_midi_ddsp"},
        ("/b", 2): {"method": "midi-ddsp", "backend": "pending_midi_ddsp"},
        ("/b", 3): {"method": "midi-ddsp", "backend": "pending_midi_ddsp"},
    }
    fs4, fs4_n = _work_for_pass(
        df, [0, 1], "fluidsynth", stem_recipe_index=fs_pending,
    )
    assert fs4 == [] and fs4_n == 0
    _, rem_pending = _work_for_pass(
        df, [0, 1], "midi_ddsp",
        stem_recipe_index={},
        tables_dir=tables,  # still has 2 real SF fallbacks for tracks 2,3
    )
    # 4 layout − 2 SF fallbacks = 2 neural remaining (pending does not credit midi_ddsp)
    assert rem_pending == 2

    # Layout-Fluidsynth neural-category stems are written method=midi-ddsp; they
    # must still clear n_fluidsynth so the bar does not stick at 0/N forever.
    df_g = pd.DataFrame({
        "path_output": ["/g"],
        "n_fluidsynth": [2],
        "n_ddsp_piano": [0],
        "n_midi_ddsp": [0],
    })
    fs_guitar = {
        ("/g", 0): {
            "method": "midi-ddsp", "backend": "fluidsynth",
            "reason": "soundfont_unsupported",
        },
        ("/g", 1): {
            "method": "midi-ddsp", "backend": "fluidsynth",
            "reason": "soundfont_unsupported",
        },
    }
    fs5, fs5_n = _work_for_pass(
        df_g, [0], "fluidsynth", stem_recipe_index=fs_guitar,
    )
    assert fs5 == [] and fs5_n == 0

    # Unsupported SF on a midi_ddsp song counts as a real Fluidsynth fallback.
    df_m = pd.DataFrame({
        "path_output": ["/m"],
        "n_fluidsynth": [0],
        "n_ddsp_piano": [0],
        "n_midi_ddsp": [3],
    })
    tables_m = tmp_path / "final_m"
    tables_m.mkdir()
    pd.DataFrame({
        "path": ["/m", "/m"],
        "track": [0, 1],
        "category": ["strings", "strings"],
        "ablation": ["ddsp_basic", "ddsp_basic"],
        "method": ["midi-ddsp", "midi-ddsp"],
        "fallback": ["basic", "basic"],
        "backend": ["fluidsynth", "fluidsynth"],
        "realify": [False, False],
        "reason": ["soundfont_unsupported", "soundfont_polyphonic"],
    }).to_csv(tables_m / "stem_recipe.fluidsynth.csv", index=False)
    _, rem_m = _work_for_pass(
        df_m, [0], "midi_ddsp", stem_recipe_index={}, tables_dir=tables_m,
    )
    # Both SF rows credit → 3 - 2 = 1 remaining.
    assert rem_m == 1


def test_merge_filters_pending_midi_ddsp(tmp_path: Path):
    from synthesis.pass_tables import merge_pass_tables, pass_recipe_csv, pass_stems_csv
    from synthesis.recipe import STEM_RECIPE_FILE_NAME

    tables = tmp_path / "final"
    tables.mkdir()
    song = "/out/SPDMX/raw/7/19/QmSong"
    pd.DataFrame({
        "song_id": ["7/19/QmSong"],
        "n_tracks": [2],
    }).to_csv(tables / MIDI_INDEX_FILE_NAME, index=False)
    pd.DataFrame([
        {
            "path": song, "track": 0, "category": "piano", "ablation": "basic",
            "method": "basic", "fallback": "basic", "backend": "fluidsynth",
            "realify": False, "reason": None,
        },
        {
            "path": song, "track": 1, "category": "strings", "ablation": "ddsp_basic",
            "method": "midi-ddsp", "fallback": "basic", "backend": "pending_midi_ddsp",
            "realify": False, "reason": "midi_ddsp_eligible",
        },
    ]).to_csv(pass_recipe_csv(tables, "fluidsynth"), index=False)
    pd.DataFrame([{
        "path": song, "track": 1, "category": "strings", "ablation": "ddsp_basic",
        "method": "midi-ddsp", "fallback": "basic", "backend": "midi_ddsp",
        "realify": False, "reason": "midi_ddsp_eligible",
    }]).to_csv(pass_recipe_csv(tables, "midi_ddsp"), index=False)
    pd.DataFrame([
        {
            "path": song, "track": 0, "original_track": 0, "program": 0,
            "is_drum": False, "name": "fs", "has_lyrics": False,
            "max_velocity": 64, "velocity_scale": 0.5,
        },
        {
            "path": song, "track": 1, "original_track": 1, "program": 1,
            "is_drum": False, "name": "md", "has_lyrics": False,
            "max_velocity": 64, "velocity_scale": 0.5,
        },
    ]).to_csv(pass_stems_csv(tables, "fluidsynth"), index=False)

    merge_pass_tables(tables)
    recipes = pd.read_csv(tables / STEM_RECIPE_FILE_NAME)
    assert set(recipes["backend"]) == {"fluidsynth", "midi_ddsp"}
    assert "pending_midi_ddsp" not in set(recipes["backend"].astype(str))


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
