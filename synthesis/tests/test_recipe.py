"""Tests for per-category hybrid recipe parsing, track plans, and resume conflicts."""

from __future__ import annotations

from pathlib import Path

import mido
import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import yaml

from shared.config import DATA_DIR_NAME, DEFAULT_AUDIO_FORMAT
from shared.csv_tables import append_rows_deduped
from synthesis.ddsp.routing import BACKEND_MIDI_DDSP, BACKEND_SOUNDFONT, route_stem
from synthesis.patches import LISTENING_CATEGORY_GM_CLASSES
from synthesis.recipe import (
    DEFAULT_RECIPE_PATH,
    METHOD_BASIC,
    METHOD_MIDI_DDSP,
    METHOD_SLAKH,
    STEM_RECIPE_COLUMNS,
    confirm_recipe_conflicts,
    hybrid_pass_for_track,
    load_recipe,
    load_stem_recipe_index,
    parse_ablation_id,
    parse_category_spec,
    require_recipe_conflicts_ok,
    resolve_track_backend,
    scan_recipe_conflicts,
)


def _all_basic() -> dict[str, str]:
    return {c: "basic" for c in LISTENING_CATEGORY_GM_CLASSES}


def _leaderboard() -> dict[str, str]:
    return {
        "piano": "basic",
        "drums": "basic",
        "guitar": "basic",
        "polyphonic": "basic",
        "voice": "basic",
        "mallet": "slakh",
        "organ": "basic_realify",
        "brass": "ddsp_basic",
        "strings": "ddsp_basic",
        "wind": "ddsp_slakh_realify",
    }


def _mono_violin():
    track = mido.MidiTrack()
    track.append(mido.Message("program_change", program=40, time=0))
    track.append(mido.Message("note_on", note=60, velocity=80, time=0))
    track.append(mido.Message("note_off", note=60, velocity=0, time=480))
    track.append(mido.Message("note_on", note=62, velocity=80, time=0))
    track.append(mido.Message("note_off", note=62, velocity=0, time=480))
    return track


def _poly_violin():
    track = mido.MidiTrack()
    track.append(mido.Message("program_change", program=40, time=0))
    track.append(mido.Message("note_on", note=60, velocity=80, time=0))
    track.append(mido.Message("note_on", note=64, velocity=80, time=0))
    track.append(mido.Message("note_off", note=60, velocity=0, time=480))
    track.append(mido.Message("note_off", note=64, velocity=0, time=0))
    return track


def test_parse_ablation_ids():
    basic = parse_ablation_id("basic")
    assert basic.method == METHOD_BASIC and not basic.realify and basic.fallback == "basic"
    a2 = parse_ablation_id("basic_realify")
    assert a2.method == METHOD_BASIC and a2.realify
    slakh = parse_ablation_id("slakh")
    assert slakh.method == METHOD_SLAKH and slakh.fallback == "slakh"
    ca1 = parse_ablation_id("ddsp_basic")
    assert ca1.method == METHOD_MIDI_DDSP and ca1.fallback == "basic" and not ca1.realify
    cb2 = parse_ablation_id("ddsp_slakh_realify")
    assert cb2.method == METHOD_MIDI_DDSP and cb2.fallback == "slakh" and cb2.realify


def test_parse_ablation_id_unknown():
    with pytest.raises(ValueError, match="Unknown ablation id"):
        parse_ablation_id("not_a_condition")


def test_parse_expanded_mapping():
    spec = parse_category_spec(
        {"method": "midi-ddsp", "realify": True, "fallback": "slakh"},
        category="wind",
    )
    assert spec.method == METHOD_MIDI_DDSP
    assert spec.realify
    assert spec.fallback == "slakh"


def test_load_recipe_from_mapping_and_categories_wrapper():
    recipe = load_recipe(_leaderboard())
    grouped = recipe.pass_categories()
    assert set(grouped["fluidsynth"]) == {
        "piano", "drums", "guitar", "polyphonic", "voice", "mallet", "organ",
    }
    assert set(grouped["ddsp"]) == {"brass", "strings", "wind"}
    assert set(grouped["realify"]) == {"organ", "wind"}
    nested = load_recipe({"categories": _all_basic()})
    assert not nested.uses_ddsp()
    assert not nested.uses_realify()


def test_load_recipe_missing_and_extra(tmp_path: Path):
    path = tmp_path / "recipe.yaml"
    path.write_text("piano: basic\n")
    with pytest.raises(ValueError, match="missing listening categories"):
        load_recipe(path)
    extra = _all_basic()
    extra["saxophone"] = "basic"
    with pytest.raises(ValueError, match="unknown categories"):
        load_recipe(extra)


def test_default_recipe_yaml_loads():
    recipe = load_recipe(DEFAULT_RECIPE_PATH)
    assert set(recipe.specs) == set(LISTENING_CATEGORY_GM_CLASSES)
    assert recipe.specs["wind"].method == METHOD_MIDI_DDSP
    assert not recipe.specs["wind"].realify
    assert recipe.specs["wind"].fallback == "slakh"
    assert recipe.specs["piano"].method == METHOD_SLAKH
    assert recipe.specs["mallet"].method == METHOD_SLAKH
    assert recipe.specs["organ"].method == METHOD_BASIC
    assert not recipe.specs["organ"].realify


def test_load_expanded_yaml(tmp_path: Path):
    doc = _all_basic()
    doc["wind"] = {"method": "midi-ddsp", "realify": True, "fallback": "slakh"}
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(doc))
    recipe = load_recipe(path)
    spec = recipe.specs["wind"]
    assert spec.method == METHOD_MIDI_DDSP and spec.realify and spec.fallback == "slakh"


def test_plan_drums_never_ddsp():
    recipe = load_recipe(_leaderboard())
    plan = recipe.plan_for_track(program=0, is_drum=True, track_name="Drums")
    assert plan.category == "drums"
    assert not plan.neural_ok
    assert resolve_track_backend(plan, route_stem(program=0, is_drum=True, check_monophony=False)) == "fluidsynth"


def test_plan_monophonic_violin_midi_ddsp():
    recipe = load_recipe(_leaderboard())
    plan = recipe.plan_for_track(program=40, is_drum=False, track_name="Violin")
    assert plan.category == "strings"
    assert plan.neural_ok
    assert not plan.use_slakh
    route = route_stem(
        program=40, is_drum=False, track=_mono_violin(), check_monophony=True,
    )
    assert route.backend == BACKEND_MIDI_DDSP
    assert resolve_track_backend(plan, route) == "midi_ddsp"


def test_hybrid_pass_for_track_matches_recipe_backend():
    from synthesis.recipe import hybrid_pass_for_track

    recipe = load_recipe(_leaderboard())
    assert hybrid_pass_for_track(
        recipe, program=0, is_drum=False, track_name="Piano",
    ) == "fluidsynth"
    assert hybrid_pass_for_track(
        recipe, program=40, is_drum=False, track_name="Violin",
    ) == "midi_ddsp"
    assert hybrid_pass_for_track(
        recipe, program=0, is_drum=True, track_name="Drums",
    ) == "fluidsynth"


def test_default_recipe_has_no_ddsp_piano_pass():
    recipe = load_recipe(DEFAULT_RECIPE_PATH)
    assert recipe.uses_ddsp()
    assert not recipe.uses_ddsp_piano()
    assert hybrid_pass_for_track(
        recipe, program=40, is_drum=False, track_name="Piano Violin",
    ) == "fluidsynth"


def test_plan_polyphonic_violin_fluidsynth_fallback():
    recipe = load_recipe(_leaderboard())
    plan = recipe.plan_for_track(program=40, is_drum=False, track_name="Violin")
    route = route_stem(
        program=40, is_drum=False, track=_poly_violin(), check_monophony=True,
    )
    assert route.backend == BACKEND_SOUNDFONT
    assert resolve_track_backend(plan, route) == "fluidsynth"
    assert plan.fallback == "basic"


def test_plan_wind_uses_slakh_and_realify():
    recipe = load_recipe(_leaderboard())
    plan = recipe.plan_for_track(program=73, is_drum=False, track_name="Flute")
    assert plan.category == "wind"
    assert plan.neural_ok
    assert plan.use_slakh
    assert plan.realify
    assert plan.fallback == "slakh"


def test_piano_basic_not_ddsp_piano():
    recipe = load_recipe(_leaderboard())
    plan = recipe.plan_for_track(program=0, is_drum=False, track_name="Piano")
    assert plan.category == "piano"
    assert not plan.neural_ok
    route = route_stem(program=0, is_drum=False, check_monophony=False)
    assert resolve_track_backend(plan, route) == "fluidsynth"


def _write_stem_tree(root: Path, *, method: str = "basic", backend: str = "fluidsynth"):
    song_dir = root / "data" / "song"
    song_dir.mkdir(parents=True)
    sf.write(str(song_dir / "stem_0.mp3"), np.zeros(44100), 44100, format="MP3")
    pd.DataFrame({"path": [str(song_dir)], "n_tracks": [1]}).to_csv(
        root / f"{DATA_DIR_NAME}.csv", index=False,
    )
    pd.DataFrame({
        "path": [str(song_dir)],
        "track": [0],
        "program": [0],
        "is_drum": [False],
        "name": ["Piano"],
    }).to_csv(root / "stems.csv", index=False)
    row = {
        "path": str(song_dir),
        "track": 0,
        "category": "piano",
        "ablation": method,
        "method": method,
        "fallback": "basic" if method != "slakh" else "slakh",
        "backend": backend,
        "realify": False,
        "reason": None,
    }
    append_rows_deduped(
        str(root / "stem_recipe.csv"),
        STEM_RECIPE_COLUMNS,
        [row],
        key_cols=["path", "track"],
    )
    return song_dir


def test_scan_recipe_conflicts_when_method_changes(tmp_path: Path):
    root = tmp_path / "stems"
    _write_stem_tree(root, method="basic")
    recipe = load_recipe({**_all_basic(), "piano": "slakh"})
    conflicts = scan_recipe_conflicts(
        root, recipe, audio_format=DEFAULT_AUDIO_FORMAT, stage="raw",
    )
    assert len(conflicts) == 1
    assert conflicts[0].category == "piano"
    assert "method=basic" in conflicts[0].recorded
    assert "method=slakh" in conflicts[0].desired


def test_scan_recipe_conflicts_empty_when_current(tmp_path: Path):
    root = tmp_path / "stems"
    _write_stem_tree(root, method="basic")
    recipe = load_recipe(_all_basic())
    assert scan_recipe_conflicts(
        root, recipe, audio_format=DEFAULT_AUDIO_FORMAT, stage="raw",
    ) == []


def test_recipe_sidecar_upsert_clears_method_conflict(tmp_path: Path):
    root = tmp_path / "stems"
    song_dir = _write_stem_tree(root, method="basic")
    recipe = load_recipe({**_all_basic(), "piano": "slakh"})
    assert scan_recipe_conflicts(
        root, recipe, audio_format=DEFAULT_AUDIO_FORMAT, stage="raw",
    )
    spec = recipe.spec_for_category("piano")
    append_rows_deduped(
        str(root / "stem_recipe.csv"),
        STEM_RECIPE_COLUMNS,
        [{
            "path": str(song_dir),
            "track": 0,
            "category": "piano",
            "ablation": spec.ablation,
            "method": spec.method,
            "fallback": spec.fallback,
            "backend": "fluidsynth",
            "realify": False,
            "reason": None,
        }],
        key_cols=["path", "track"],
    )
    assert scan_recipe_conflicts(
        root, recipe, audio_format=DEFAULT_AUDIO_FORMAT, stage="raw",
    ) == []


def test_confirm_recipe_conflicts_yes_flag():
    from synthesis.recipe import RecipeConflict

    conflicts = [RecipeConflict("p", 0, "piano", "old", "new")]
    assert confirm_recipe_conflicts(conflicts, yes=True)
    assert not confirm_recipe_conflicts(conflicts, yes=False, input_fn=lambda _: "n")
    assert confirm_recipe_conflicts(conflicts, yes=False, input_fn=lambda _: "y")


def test_require_recipe_conflicts_ok_aborts():
    from synthesis.recipe import RecipeConflict

    with pytest.raises(SystemExit, match="Aborted"):
        require_recipe_conflicts_ok(
            [RecipeConflict("p", 0, "piano", "old", "new")],
            yes=False,
            input_fn=lambda _: "n",
        )


def test_load_stem_recipe_index_tolerates_empty_file(tmp_path: Path):
    empty = tmp_path / "stem_recipe.midi_ddsp.csv"
    empty.write_text("")
    assert load_stem_recipe_index(tmp_path, filename=empty.name) == {}
    empty.write_bytes(b"")
    assert load_stem_recipe_index(tmp_path, filename=empty.name) == {}
    missing = tmp_path / "missing.csv"
    assert load_stem_recipe_index(tmp_path, filename=missing.name) == {}


def test_load_stem_recipe_index_tolerates_reason_schema_drift(tmp_path: Path):
    """8-col header + 9-col data rows must not wipe the resume index."""
    path = tmp_path / "stem_recipe.midi_ddsp.csv"
    path.write_text(
        "path,track,category,ablation,method,fallback,backend,realify\n"
        "/song,0,strings,ddsp_basic,midi-ddsp,basic,midi_ddsp,False\n"
        "/song,1,wind,ddsp_slakh,midi-ddsp,slakh,midi_ddsp,False,midi_ddsp_eligible\n",
        encoding="utf-8",
    )
    idx = load_stem_recipe_index(tmp_path, filename=path.name)
    assert len(idx) == 2
    assert idx[("/song", 1)]["backend"] == "midi_ddsp"
    assert idx[("/song", 1)].get("reason") == "midi_ddsp_eligible"
