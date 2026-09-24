"""Tests for sweep response aggregation."""

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from experiments.listening.aggregate import (
    aggregate_winners,
    noise_level_winners,
    ratings_dataframe,
    run_aggregate,
    swipe_winners,
)
from experiments.listening.catalog import SweepCatalog


def _write_catalog_tree(tmp_path: Path) -> Path:
    sweep_dir = tmp_path / "sweep"
    song_dir = tmp_path / "basic" / "data" / "0/13/QmTest"
    song_dir.mkdir(parents=True)
    (song_dir / "stem_0.flac").write_bytes(b"x")

    variant_dir = sweep_dir / "variants" / "noise0.45_minimal" / "data" / "0/13/QmTest"
    variant_dir.mkdir(parents=True)
    (variant_dir / "stem_0.flac").write_bytes(b"x")

    variant_dir2 = sweep_dir / "variants" / "noise0.55_minimal" / "data" / "0/13/QmTest"
    variant_dir2.mkdir(parents=True)
    (variant_dir2 / "stem_0.flac").write_bytes(b"x")

    pd.DataFrame([
        {
            "variant_id": "noise0.45_minimal",
            "init_noise_level": 0.45,
            "prompt_variant": "minimal",
            "prompt": "piano",
            "stem_id": "piano_test",
            "category": "piano",
            "path": str(song_dir),
            "track": 0,
            "out_path": str(variant_dir / "stem_0.flac"),
        },
        {
            "variant_id": "noise0.55_minimal",
            "init_noise_level": 0.55,
            "prompt_variant": "minimal",
            "prompt": "piano",
            "stem_id": "piano_test",
            "category": "piano",
            "path": str(song_dir),
            "track": 0,
            "out_path": str(variant_dir2 / "stem_0.flac"),
        },
    ]).to_csv(sweep_dir / "manifest.csv", index=False)

    probe_path = tmp_path / "probe_stems.yaml"
    probe_path.write_text(yaml.dump({
        "stems": [{
            "id": "piano_test",
            "category": "piano",
            "song_id": "0/13/QmTest",
            "track": 0,
        }],
    }))
    return sweep_dir


def test_ratings_dataframe():
    responses = {
        "ratings": [{
            "stem_id": "piano_test",
            "category": "piano",
            "samples": [
                {"variant_id": "noise0.45_minimal", "content": 5, "realism": 4},
                {"variant_id": "noise0.55_minimal", "content": 3, "realism": 5},
            ],
        }],
    }
    df = ratings_dataframe(responses)
    assert len(df) == 2


def test_aggregate_winners_prefers_realism_when_content_ok():
    responses = {
        "ratings": [{
            "stem_id": "piano_test",
            "category": "piano",
            "samples": [
                {"variant_id": "noise0.45_minimal", "content": 5, "realism": 4},
                {"variant_id": "noise0.55_minimal", "content": 4, "realism": 5},
            ],
        }],
    }
    df = ratings_dataframe(responses)
    _, winners = aggregate_winners(df)
    assert winners.iloc[0]["variant_id"] == "noise0.55_minimal"


def test_aggregate_filters_low_content(tmp_path: Path, capsys):
    sweep_dir = _write_catalog_tree(tmp_path)
    responses_path = tmp_path / "responses.json"
    responses_path.write_text(json.dumps({
        "ratings": [{
            "stem_id": "piano_test",
            "category": "piano",
            "samples": [
                {"variant_id": "noise0.45_minimal", "content": 5, "realism": 3},
                {"variant_id": "noise0.55_minimal", "content": 2, "realism": 5},
            ],
        }],
    }))

    output_path = tmp_path / "results.md"
    with pytest.raises(RuntimeError, match="preset_sweep"):
        run_aggregate(
            sweep_type="preset",
            responses_path=responses_path,
            output_path=output_path,
            sweep_dir=sweep_dir,
            content_threshold=3.0,
            content_mean_threshold=3.5,
        )


def test_noise_level_winners_requires_content_then_picks_realism():
    responses = {
        "ratings": [{
            "stem_id": "piano_a",
            "category": "piano",
            "samples": [
                {"variant_id": "noise0.45", "content": 5.0, "realism": 4.0},
                {"variant_id": "noise0.55", "content": 4.0, "realism": 5.0},
                {"variant_id": "noise0.65", "content": 3.0, "realism": 5.0},
            ],
        }, {
            "stem_id": "piano_b",
            "category": "piano",
            "samples": [
                {"variant_id": "noise0.45", "content": 5.0, "realism": 4.0},
                {"variant_id": "noise0.55", "content": 4.0, "realism": 5.0},
                {"variant_id": "noise0.65", "content": 3.0, "realism": 5.0},
            ],
        }],
    }
    df = ratings_dataframe(responses)
    winners = noise_level_winners(df, content_threshold=4.5)
    assert winners.iloc[0]["variant_id"] == "noise0.45"
    assert bool(winners.iloc[0]["passed_content_threshold"])


def test_noise_level_winners_excludes_low_content_even_if_more_realistic():
    responses = {
        "ratings": [{
            "stem_id": "piano_a",
            "category": "piano",
            "samples": [
                {"variant_id": "noise0.45", "content": 4.8, "realism": 4.0},
                {"variant_id": "noise0.55", "content": 4.0, "realism": 5.0},
            ],
        }, {
            "stem_id": "piano_b",
            "category": "piano",
            "samples": [
                {"variant_id": "noise0.45", "content": 4.8, "realism": 4.0},
                {"variant_id": "noise0.55", "content": 4.0, "realism": 5.0},
            ],
        }],
    }
    df = ratings_dataframe(responses)
    winners = noise_level_winners(df, content_threshold=4.5)
    assert winners.iloc[0]["variant_id"] == "noise0.45"


def test_noise_level_winners_tiebreaks_on_higher_noise():
    responses = {
        "ratings": [{
            "stem_id": "piano_a",
            "category": "piano",
            "samples": [
                {"variant_id": "noise0.45", "content": 5.0, "realism": 4.5},
                {"variant_id": "noise0.55", "content": 5.0, "realism": 4.5},
            ],
        }, {
            "stem_id": "piano_b",
            "category": "piano",
            "samples": [
                {"variant_id": "noise0.45", "content": 5.0, "realism": 4.5},
                {"variant_id": "noise0.55", "content": 5.0, "realism": 4.5},
            ],
        }],
    }
    df = ratings_dataframe(responses)
    winners = noise_level_winners(df, content_threshold=4.5)
    assert winners.iloc[0]["variant_id"] == "noise0.55"


def test_noise_level_winners_falls_back_when_none_pass_content():
    responses = {
        "ratings": [{
            "stem_id": "piano_a",
            "category": "piano",
            "samples": [
                {"variant_id": "noise0.45", "content": 4.0, "realism": 4.0},
                {"variant_id": "noise0.55", "content": 3.0, "realism": 5.0},
            ],
        }],
    }
    df = ratings_dataframe(responses)
    winners = noise_level_winners(df, content_threshold=4.5)
    assert winners.iloc[0]["variant_id"] == "noise0.55"
    assert not bool(winners.iloc[0]["passed_content_threshold"])


def _swipe_responses(votes: list[dict]) -> dict:
    return {"mode": "swipe", "votes": votes}


def test_swipe_winners_includes_all_strong_accepts():
    responses = _swipe_responses([
        {"category": "piano", "variant_id": "sf_a", "clip_id": "c0", "tier": "strong_accept"},
        {"category": "piano", "variant_id": "sf_b", "clip_id": "c0", "tier": "weak_accept"},
        {"category": "piano", "variant_id": "sf_c", "clip_id": "c0", "tier": "strong_accept"},
    ])
    winners = swipe_winners(responses)
    assert winners["piano"] == ["sf_a", "sf_c"]


def test_swipe_winners_weak_accept_fallback_respects_min_winners():
    responses = _swipe_responses([
        {"category": "piano", "variant_id": "sf_a", "clip_id": "c0", "tier": "weak_accept"},
        {"category": "piano", "variant_id": "sf_a", "clip_id": "c1", "tier": "weak_accept"},
        {"category": "piano", "variant_id": "sf_b", "clip_id": "c0", "tier": "weak_accept"},
        {"category": "piano", "variant_id": "sf_c", "clip_id": "c0", "tier": "weak_accept"},
        {"category": "piano", "variant_id": "sf_d", "clip_id": "c0", "tier": "weak_accept"},
    ])
    winners = swipe_winners(responses, min_winners=3)
    assert winners["piano"] == ["sf_a", "sf_b", "sf_c"]


def test_swipe_winners_strict_raises_on_reject_only():
    responses = _swipe_responses([
        {"category": "piano", "variant_id": "sf_a", "clip_id": "c0", "tier": "weak_reject"},
    ])
    with pytest.raises(RuntimeError, match="reject-tier fallback"):
        swipe_winners(responses, strict=True)


def test_swipe_winners_allow_reject_fallback():
    responses = _swipe_responses([
        {"category": "piano", "variant_id": "sf_a", "clip_id": "c0", "tier": "weak_reject"},
        {"category": "piano", "variant_id": "sf_b", "clip_id": "c0", "tier": "strong_reject"},
    ])
    winners = swipe_winners(responses, strict=False, min_winners=1)
    assert winners["piano"] == ["sf_a"]


def test_swipe_winners_max_winners_caps_strong_accept_pool():
    responses = _swipe_responses([
        {"category": "piano", "variant_id": "fx_dry", "clip_id": "c0", "tier": "strong_accept"},
        {"category": "piano", "variant_id": "fx_light", "clip_id": "c0", "tier": "strong_accept"},
        {"category": "piano", "variant_id": "fx_light", "clip_id": "c1", "tier": "strong_accept"},
    ])
    winners = swipe_winners(responses, max_winners=1)
    assert winners["piano"] == ["fx_light"]

    all_winners = swipe_winners(responses)
    assert set(all_winners["piano"]) == {"fx_dry", "fx_light"}


def test_swipe_winners_excludes_legacy_polyphonic_stems():
    responses = _swipe_responses([
        {"category": "polyphonic", "variant_id": "sf_a", "stem_id": "poly_flute", "clip_id": "c0", "tier": "strong_accept"},
        {"category": "polyphonic", "variant_id": "sf_b", "stem_id": "synth_lead", "clip_id": "c0", "tier": "strong_accept"},
    ])
    winners = swipe_winners(responses)
    assert winners["polyphonic"] == ["sf_b"]
