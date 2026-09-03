"""Tests for CSV table helpers."""

import warnings

import pandas as pd

from shared.config import STEMS_TABLE_COLUMNS
from shared.csv_tables import append_rows_deduped, sanitize_track_name


def test_sanitize_track_name_strips_null_bytes():
    assert sanitize_track_name("Piano\x00") == "Piano"
    assert sanitize_track_name("  a\x00b  ") == "ab"
    assert sanitize_track_name("") is None
    assert sanitize_track_name(None) is None


def test_append_rows_does_not_rewrite_existing(tmp_path):
    from shared.csv_tables import append_rows

    csv_path = str(tmp_path / "stems.fluidsynth.csv")
    row = {
        "path": "/songs/a", "track": 0, "original_track": 0, "program": 0,
        "is_drum": False, "name": "Piano", "has_lyrics": False,
        "max_velocity": 64, "velocity_scale": 0.5,
    }
    append_rows(csv_path, STEMS_TABLE_COLUMNS, [row])
    append_rows(csv_path, STEMS_TABLE_COLUMNS, [{**row, "track": 1}])
    stems = pd.read_csv(csv_path)
    assert len(stems) == 2
    assert set(stems["track"].astype(int)) == {0, 1}


def test_append_rows_deduped_replaces_path(tmp_path):
    csv_path = tmp_path / "stems.csv"
    path = "/songs/a"
    rows_a = [
        {
            "path": path, "track": 0, "original_track": 0, "program": 0,
            "is_drum": False, "name": sanitize_track_name("Piano\x00"), "has_lyrics": False,
            "max_velocity": 64, "velocity_scale": 0.5,
        },
        {
            "path": path, "track": 1, "original_track": 1, "program": 1,
            "is_drum": False, "name": "Bass", "has_lyrics": False,
            "max_velocity": 127, "velocity_scale": 1.0,
        },
    ]
    append_rows_deduped(str(csv_path), STEMS_TABLE_COLUMNS, rows_a)
    append_rows_deduped(
        str(csv_path),
        STEMS_TABLE_COLUMNS,
        [{**rows_a[0], "name": "Piano"}, rows_a[1]],
    )
    stems = pd.read_csv(csv_path)
    assert len(stems) == 2
    assert stems[stems["track"] == 0].iloc[0]["name"] == "Piano"


def test_append_rows_deduped_composite_key(tmp_path):
    csv_path = tmp_path / "stem_recipe.csv"
    path = "/songs/a"
    append_rows_deduped(
        str(csv_path),
        ["path", "track", "method"],
        [{"path": path, "track": 0, "method": "basic"}, {"path": path, "track": 1, "method": "slakh"}],
        key_cols=["path", "track"],
    )
    append_rows_deduped(
        str(csv_path),
        ["path", "track", "method"],
        [{"path": path, "track": 0, "method": "midi-ddsp"}],
        key_cols=["path", "track"],
    )
    df = pd.read_csv(csv_path)
    assert len(df) == 2
    assert df[df["track"] == 0].iloc[0]["method"] == "midi-ddsp"
    assert df[df["track"] == 1].iloc[0]["method"] == "slakh"


def test_append_rows_deduped_stem_track_keeps_siblings(tmp_path):
    csv_path = tmp_path / "stems.csv"
    path = "/songs/a"
    rows = [
        {
            "path": path, "track": 0, "original_track": 0, "program": 0,
            "is_drum": False, "name": "Piano", "has_lyrics": False,
            "max_velocity": 64, "velocity_scale": 0.5,
        },
        {
            "path": path, "track": 1, "original_track": 1, "program": 48,
            "is_drum": False, "name": "Strings", "has_lyrics": False,
            "max_velocity": 90, "velocity_scale": 0.7,
        },
    ]
    append_rows_deduped(str(csv_path), STEMS_TABLE_COLUMNS, rows, key_cols=["path", "track"])
    append_rows_deduped(
        str(csv_path),
        STEMS_TABLE_COLUMNS,
        [{**rows[0], "program": 1, "name": "Slakh Piano"}],
        key_cols=["path", "track"],
    )
    stems = pd.read_csv(csv_path)
    assert len(stems) == 2
    assert stems[stems["track"] == 0].iloc[0]["name"] == "Slakh Piano"
    assert int(stems[stems["track"] == 0].iloc[0]["program"]) == 1
    assert stems[stems["track"] == 1].iloc[0]["name"] == "Strings"


def test_append_rows_deduped_all_na_column_no_future_warning(tmp_path):
    csv_path = tmp_path / "stems.csv"
    unnamed = {
        "path": "/songs/a", "track": 0, "original_track": 0, "program": 0,
        "is_drum": False, "name": None, "has_lyrics": False,
        "max_velocity": 64, "velocity_scale": 0.5,
    }
    named = {
        "path": "/songs/b", "track": 0, "original_track": 0, "program": 0,
        "is_drum": False, "name": "Piano", "has_lyrics": False,
        "max_velocity": 64, "velocity_scale": 0.5,
    }
    append_rows_deduped(str(csv_path), STEMS_TABLE_COLUMNS, [unnamed], key_cols=["path", "track"])
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        append_rows_deduped(
            str(csv_path), STEMS_TABLE_COLUMNS, [named], key_cols=["path", "track"],
        )
    stems = pd.read_csv(csv_path)
    assert len(stems) == 2


def test_append_rows_deduped_parallel_writers(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    csv_path = str(tmp_path / "stems.csv")

    def _write(track: int) -> None:
        append_rows_deduped(
            csv_path,
            STEMS_TABLE_COLUMNS,
            [{
                "path": "/songs/a", "track": track, "original_track": track, "program": track,
                "is_drum": False, "name": f"T{track}", "has_lyrics": False,
                "max_velocity": 64, "velocity_scale": 0.5,
            }],
            key_cols=["path", "track"],
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(_write, range(20)))
    stems = pd.read_csv(csv_path)
    assert len(stems) == 20
    assert set(stems["track"].astype(int)) == set(range(20))


def test_read_csv_flexible_and_append_migrates_new_column(tmp_path):
    from shared.csv_tables import append_rows, read_csv_flexible

    csv_path = tmp_path / "stem_recipe.midi_ddsp.csv"
    old_cols = [
        "path", "track", "category", "ablation", "method", "fallback", "backend", "realify",
    ]
    new_cols = old_cols + ["reason"]
    csv_path.write_text(
        ",".join(old_cols) + "\n"
        + "/song,0,strings,ddsp_basic,midi-ddsp,basic,midi_ddsp,False\n"
        + "/song,1,wind,ddsp_slakh,midi-ddsp,slakh,midi_ddsp,False,midi_ddsp_eligible\n",
        encoding="utf-8",
    )
    df = read_csv_flexible(csv_path, columns=new_cols)
    assert len(df) == 2
    assert list(df.columns) == new_cols
    assert str(df.loc[df.track.astype(str) == "1", "reason"].iloc[0]) == "midi_ddsp_eligible"

    append_rows(str(csv_path), new_cols, [{
        "path": "/song", "track": 2, "category": "brass", "ablation": "ddsp_basic",
        "method": "midi-ddsp", "fallback": "basic", "backend": "midi_ddsp",
        "realify": False, "reason": "midi_ddsp_eligible",
    }])
    rewritten = pd.read_csv(csv_path)
    assert list(rewritten.columns) == new_cols
    assert len(rewritten) == 3
