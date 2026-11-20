"""Tests for release index export / soundfont provenance table."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from experiments.patch_sweep.winners import pick_soundfont_id, soundfont_seed
from synthesis.export_release_indexes import (
    build_stem_soundfonts,
    export_release_indexes,
)


def test_soundfont_seed_is_process_stable():
    a = soundfont_seed(43, "/song/a", "voice")
    b = soundfont_seed(43, "/song/a", "voice")
    assert a == b
    assert a == soundfont_seed(43, "/song/a", "voice")


def test_pick_soundfont_stable_across_calls():
    ids = ["choriumreva", "dsoundfontv4", "sgm_v2_01"]
    assert pick_soundfont_id(
        ids, category="voice", song_path="/raw/1/2/QmX", sample_seed=43,
    ) == pick_soundfont_id(
        ids, category="voice", song_path="/raw/1/2/QmX", sample_seed=43,
    )


def test_build_stem_soundfonts_provenance(tmp_path: Path):
    stems = pd.DataFrame({
        "song_id": ["1/2/QmA", "1/2/QmA", "1/2/QmB"],
        "track": [0, 1, 0],
    })
    recipe = pd.DataFrame({
        "path": [
            "/data/SPDMX_dev/raw/1/2/QmA",
            "/data/SPDMX_dev/raw/1/2/QmA",
            "/data/SPDMX_dev/raw/1/2/QmB",
        ],
        "track": [0, 1, 0],
        "category": ["piano", "voice", "strings"],
        "ablation": ["basic", "slakh", "ddsp_basic"],
        "method": ["basic", "slakh", "midi-ddsp"],
        "fallback": ["basic", "slakh", "basic"],
        "backend": ["fluidsynth", "fluidsynth", "midi_ddsp"],
        "realify": [False, False, False],
        "reason": ["", "", "midi_ddsp_eligible"],
    })
    out = build_stem_soundfonts(stems, recipe)
    assert list(out["provenance"]) == [
        "basic_default",
        "slakh_unlogged",
        "neural",
    ]
    assert pd.notna(out.loc[0, "soundfont_id"]) and out.loc[0, "soundfont_id"]
    assert pd.isna(out.loc[1, "soundfont_id"])
    assert pd.isna(out.loc[2, "soundfont_id"])
    assert out.loc[1, "soundfont_id_locked"] == "choriumreva"


def test_export_writes_gz(tmp_path: Path):
    root = tmp_path / "SPDMX"
    root.mkdir()
    pd.DataFrame({
        "song_id": ["1/2/QmA"],
        "track": [0],
        "path": ["./chunk_0/1/2/QmA"],
        "mid": ["./chunk_0/1/2/QmA/mix.mid"],
        "mix": ["./chunk_0/1/2/QmA/mix.flac"],
        "original_track": [0],
        "program": [0],
        "is_drum": [False],
        "name": [""],
        "chunk": [0],
    }).to_csv(root / "stems.csv", index=False)
    pd.DataFrame({
        "song_id": ["1/2/QmA"],
        "path": ["./chunk_0/1/2/QmA"],
        "mid": ["./chunk_0/1/2/QmA/mix.mid"],
        "mix": ["./chunk_0/1/2/QmA/mix.flac"],
        "chunk": [0],
        "n_tracks": [1],
        "subset:all": [True],
    }).to_csv(root / "songs.csv", index=False)

    recipe = tmp_path / "stem_recipe.csv"
    pd.DataFrame({
        "path": ["/x/raw/1/2/QmA"],
        "track": [0],
        "category": ["piano"],
        "ablation": ["basic"],
        "method": ["basic"],
        "fallback": ["basic"],
        "backend": ["fluidsynth"],
        "realify": [False],
        "reason": [""],
        "soundfont_id": ["sgm_v2"],
        "fx_profile": ["dry"],
    }).to_csv(recipe, index=False)

    out_dir = tmp_path / "indexes"
    written = export_release_indexes(
        spdmx_root=root,
        out_dir=out_dir,
        recipe_path=recipe,
    )
    assert (out_dir / "stems.csv.gz").is_file()
    assert (out_dir / "songs.csv.gz").is_file()
    assert (out_dir / "stem_soundfonts.csv.gz").is_file()
    sf = pd.read_csv(written["stem_soundfonts.csv"])
    assert sf.iloc[0]["provenance"] == "recorded"
    assert sf.iloc[0]["soundfont_id"] == "sgm_v2"
