"""Tests for verification listening workflow."""

import json
from pathlib import Path

import pytest

import pandas as pd

from experiments.listening.verification import (
    PRESET_VERIFY_SOURCE,
    analyze_responses,
    build_preset_realify_verification_meta,
    build_verification_meta,
    validate_verification,
    winners_from_verification,
)


def _responses():
    return {
        "ratings": [
            {
                "stem_id": "piano_a",
                "category": "piano",
                "samples": [
                    {"variant_id": "good", "content": 5, "realism": 4},
                    {"variant_id": "bad", "content": 2, "realism": 5},
                    {"variant_id": "ok", "content": 4, "realism": 3},
                ],
            },
            {
                "stem_id": "piano_b",
                "category": "piano",
                "samples": [
                    {"variant_id": "good", "content": 4, "realism": 5},
                    {"variant_id": "bad", "content": 3, "realism": 4},
                    {"variant_id": "ok", "content": 4, "realism": 4},
                ],
            },
        ],
    }


def test_analyze_responses_filters_low_content():
    _, stats, eligible, winners = analyze_responses(_responses())
    assert "bad" not in set(eligible["variant_id"])
    assert winners.iloc[0]["variant_id"] == "good"


def test_winners_from_verification():
    doc = {
        "categories": [
            {
                "category": "piano",
                "approved": ["good", "ok"],
                "winner_variant_id": "ok",
            },
        ],
    }
    assert winners_from_verification(doc) == {"piano": "ok"}


def test_winners_from_verification_patch_shortlist():
    doc = {
        "sweep_type": "patch",
        "categories": [
            {
                "category": "piano",
                "approved": ["sgm_v2", "airfont_380"],
            },
        ],
    }
    assert winners_from_verification(doc, sweep_type="patch") == {
        "piano": ["sgm_v2", "airfont_380"],
    }


def test_validate_verification_patch_requires_one_soundfont():
    errors = validate_verification({
        "sweep_type": "patch",
        "categories": [{
            "category": "piano",
            "approved": [],
        }],
    })
    assert any("keep at least one soundfont" in err for err in errors)

    errors = validate_verification({
        "sweep_type": "patch",
        "categories": [{
            "category": "piano",
            "approved": ["sgm_v2"],
        }],
    })
    assert errors == []


def test_validate_verification_requires_approved_winner():
    errors = validate_verification({
        "sweep_type": "preset",
        "categories": [{
            "category": "piano",
            "approved": [],
            "winner_variant_id": "good",
        }],
    })
    assert any("approve" in err for err in errors)

    errors = validate_verification({
        "sweep_type": "preset",
        "categories": [{
            "category": "piano",
            "approved": ["good"],
            "winner_variant_id": "ok",
        }],
    })
    assert any("winner must be among approved" in err for err in errors)


def test_bypass_routing_rules_from_verification_partial_category():
    from experiments.listening.verification import (
        bypass_realify_from_verification,
        bypass_routing_rules_from_verification,
    )

    doc = {
        "categories": [{
            "category": "voice",
            "bypass_realify": False,
            "stems": [
                {
                    "stem_id": "voice_a",
                    "track_name": "Soprano",
                    "program": 52,
                    "is_drum": False,
                    "bypass_realify": True,
                },
                {
                    "stem_id": "voice_b",
                    "track_name": "Choir",
                    "program": 52,
                    "is_drum": False,
                    "bypass_realify": False,
                },
            ],
        }],
    }
    assert bypass_realify_from_verification(doc) == {}
    with pytest.raises(RuntimeError, match="preset_sweep"):
        bypass_routing_rules_from_verification(doc)


def test_bypass_realify_from_verification_all_stems():
    from experiments.listening.verification import bypass_realify_from_verification

    doc = {
        "categories": [{
            "category": "organ",
            "bypass_realify": False,
            "stems": [
                {"stem_id": "organ_a", "track_name": "Organ", "bypass_realify": True},
                {"stem_id": "organ_b", "track_name": "Organ 2", "bypass_realify": True},
            ],
        }],
    }
    assert bypass_realify_from_verification(doc) == {"organ": True}


def test_validate_verification_preset_bypass_skips_winner():
    errors = validate_verification({
        "sweep_type": "preset",
        "categories": [{
            "category": "organ",
            "approved": [],
            "winner_variant_id": None,
            "bypass_realify": True,
        }],
    })
    assert errors == []


def test_bypass_realify_from_verification():
    from experiments.listening.verification import bypass_realify_from_verification

    doc = {
        "categories": [
            {"category": "organ", "bypass_realify": True},
            {"category": "piano", "bypass_realify": False, "winner_variant_id": "good"},
        ],
    }
    assert bypass_realify_from_verification(doc) == {"organ": True}


def test_winners_from_verification_skips_bypass():
    doc = {
        "sweep_type": "preset",
        "categories": [
            {"category": "organ", "bypass_realify": True},
            {"category": "piano", "approved": ["good"], "winner_variant_id": "good"},
        ],
    }
    assert winners_from_verification(doc, sweep_type="preset") == {"piano": "good"}


def test_build_verification_meta_marks_filter_pass(tmp_path: Path):
    from experiments.listening.catalog import SweepCatalog
    import yaml

    sweep_dir = tmp_path / "sweep"
    song_dir = tmp_path / "basic" / "data" / "0/13/QmTest"
    song_dir.mkdir(parents=True)
    (song_dir / "stem_0.flac").write_bytes(b"x")

    for variant_id in ("good", "bad", "ok"):
        variant_dir = sweep_dir / "variants" / variant_id / "data" / "0/13/QmTest"
        variant_dir.mkdir(parents=True)
        (variant_dir / "stem_0.flac").write_bytes(b"x")

    rows = []
    for stem_id in ("piano_a", "piano_b"):
        for variant_id in ("good", "bad", "ok"):
            rows.append({
                "variant_id": variant_id,
                "init_noise_level": 0.45,
                "prompt_variant": "minimal",
                "stem_id": stem_id,
                "category": "piano",
                "path": str(song_dir),
                "track": 0,
                "out_path": str(sweep_dir / "variants" / variant_id / "data" / "0/13/QmTest" / "stem_0.flac"),
            })
    pd.DataFrame(rows).to_csv(sweep_dir / "manifest.csv", index=False)

    probe_path = tmp_path / "probe_stems.yaml"
    probe_path.write_text(yaml.dump({
        "stems": [
            {"id": "piano_a", "category": "piano", "song_id": "0/13/QmTest", "track": 0},
            {"id": "piano_b", "category": "piano", "song_id": "0/13/QmTest", "track": 0},
        ],
    }))

    with pytest.raises(RuntimeError, match="preset_sweep"):
        SweepCatalog("preset", tmp_path / "sweep", tmp_path / "basic")
def test_verification_from_patch_swipe_votes():
    from experiments.listening.verification import verification_from_patch_swipe_votes

    shortlists = {
        "piano": ["sf_a", "sf_b", "sf_c"],
        "drums": ["sf_d"],
    }
    votes = [
        {"category": "piano", "variant_id": "sf_a", "tier": "strong_accept"},
        {"category": "piano", "variant_id": "sf_b", "tier": "strong_reject"},
        {"category": "drums", "variant_id": "sf_d", "tier": "strong_accept"},
    ]
    doc = verification_from_patch_swipe_votes(
        votes,
        shortlists,
        pass_number=2,
        source_verification="verification_final_winners_yaml_prev.json",
    )
    assert doc["verification_mode"] == "soundfont_shortlist_swipe"
    assert doc["pass"] == 2
    assert doc["source_verification"] == "verification_final_winners_yaml_prev.json"
    by_cat = {entry["category"]: entry["approved"] for entry in doc["categories"]}
    assert by_cat["piano"] == ["sf_a"]
    assert by_cat["drums"] == ["sf_d"]
    assert validate_verification(doc) == []


def test_shortlists_from_verification_doc_for_next_pass():
    from experiments.listening.verification import shortlists_from_verification_doc

    doc = {
        "sweep_type": "patch",
        "categories": [
            {"category": "piano", "approved": ["sf_a", "sf_b"]},
            {"category": "drums", "approved": []},
            {"category": "strings", "approved": ["sf_s"]},
        ],
    }
    assert shortlists_from_verification_doc(doc) == {
        "piano": ["sf_a", "sf_b"],
        "strings": ["sf_s"],
    }


def test_category_counts_from_votes():
    from experiments.listening.verification import category_counts_from_votes

    cards = [
        {"card_id": "p|a", "category": "piano"},
        {"card_id": "p|b", "category": "piano"},
        {"card_id": "p|c", "category": "piano"},
        {"card_id": "d|a", "category": "drums"},
    ]
    votes = [
        {"card_id": "p|a", "tier": "strong_accept"},
        {"card_id": "p|b", "tier": "strong_reject"},
    ]
    counts = category_counts_from_votes(cards, votes)
    assert counts["piano"] == {"total": 3, "left": 1, "accepted": 1, "rejected": 1}
    assert counts["drums"] == {"total": 1, "left": 1, "accepted": 0, "rejected": 0}


def test_patch_verify_swipe_session_path_is_pass_aware(tmp_path: Path):
    from experiments.listening.catalog import SweepCatalog
    from experiments.listening.verification import patch_verify_swipe_session_path
    import yaml

    sweep_dir = tmp_path / "phase1_soundfonts"
    sweep_dir.mkdir()
    (sweep_dir / "manifest.csv").write_text("variant_id\n")
    probe_path = tmp_path / "probe_stems.yaml"
    probe_path.write_text(yaml.dump({"stems": []}))
    catalog = SweepCatalog("patch", sweep_dir, tmp_path / "basic", probe_stems_path=probe_path)

    legacy = patch_verify_swipe_session_path(catalog)
    assert legacy.name == "verification_swipe_in_progress.json"

    pass2 = patch_verify_swipe_session_path(
        catalog,
        pass_number=2,
        source_verification="verification_final_winners_yaml_abc.json",
    )
    assert "pass2" in pass2.name
    assert "verification_final_winners_yaml_abc" in pass2.name
    assert pass2.name.endswith("_in_progress.json")


def test_build_patch_verify_swipe_meta_includes_pass(tmp_path: Path):
    from experiments.listening.catalog import SweepCatalog
    from experiments.listening.verification import build_patch_verify_swipe_meta
    import yaml

    sweep_dir = tmp_path / "phase1_archive_soundfonts"
    song_dir = tmp_path / "basic" / "data" / "0/13/QmTest"
    song_dir.mkdir(parents=True)
    (song_dir / "stem_0.flac").write_bytes(b"x")

    clip_path = (
        sweep_dir / "clips" / "variants" / "sf_a" / "data" / "0/13/QmTest" / "stem_0_c0.mp3"
    )
    clip_path.parent.mkdir(parents=True)
    clip_path.write_bytes(b"mp3")

    pd.DataFrame([{
        "phase": "phase1_archive_soundfonts",
        "variant_id": "sf_a",
        "soundfont_id": "sf_a",
        "stem_id": "piano_a",
        "category": "piano",
        "path": str(song_dir),
        "track": 0,
        "out_path": str(clip_path),
        "clip_id": "piano_a_c0",
        "clip_index": 0,
    }]).to_csv(sweep_dir / "clip_manifest.csv", index=False)

    pd.DataFrame([{
        "variant_id": "sf_a",
        "soundfont_id": "sf_a",
        "stem_id": "piano_a",
        "category": "piano",
        "path": str(song_dir),
        "track": 0,
        "out_path": str(sweep_dir / "variants" / "sf_a" / "data" / "0/13/QmTest" / "stem_0.flac"),
    }]).to_csv(sweep_dir / "manifest.csv", index=False)
    variant_dir = sweep_dir / "variants" / "sf_a" / "data" / "0/13/QmTest"
    variant_dir.mkdir(parents=True)
    (variant_dir / "stem_0.flac").write_bytes(b"x")

    probe_path = tmp_path / "probe_stems.yaml"
    probe_path.write_text(yaml.dump({
        "stems": [
            {"id": "piano_a", "category": "piano", "song_id": "0/13/QmTest", "track": 0},
        ],
    }))

    catalog = SweepCatalog("patch", sweep_dir, tmp_path / "basic", probe_stems_path=probe_path)
    meta = build_patch_verify_swipe_meta(
        catalog,
        {"piano": ["sf_a"]},
        pass_number=2,
        source_verification="verification_final_prev.json",
    )
    assert meta["pass"] == 2
    assert meta["source_verification"] == "verification_final_prev.json"
    assert meta["shortlists"] == {"piano": ["sf_a"]}
    assert meta["category_totals"] == {"piano": 1}
    assert "pass2" in meta["session_id"]
    assert "verify_pass2" in meta["storage_key"]


def test_build_patch_verify_swipe_cards(tmp_path: Path):
    from experiments.listening.catalog import SweepCatalog
    from experiments.listening.verification import build_patch_verify_swipe_cards
    import yaml

    sweep_dir = tmp_path / "phase1_archive_soundfonts"
    song_dir = tmp_path / "basic" / "data" / "0/13/QmTest"
    song_dir.mkdir(parents=True)
    (song_dir / "stem_0.flac").write_bytes(b"x")

    clip_path = (
        sweep_dir / "clips" / "variants" / "sf_a" / "data" / "0/13/QmTest" / "stem_0_c0.mp3"
    )
    clip_path.parent.mkdir(parents=True)
    clip_path.write_bytes(b"mp3")

    pd.DataFrame([{
        "phase": "phase1_archive_soundfonts",
        "variant_id": "sf_a",
        "soundfont_id": "sf_a",
        "stem_id": "piano_a",
        "category": "piano",
        "path": str(song_dir),
        "track": 0,
        "out_path": str(clip_path),
        "clip_id": "piano_a_c0",
        "clip_index": 0,
    }]).to_csv(sweep_dir / "clip_manifest.csv", index=False)

    pd.DataFrame([{
        "variant_id": "sf_a",
        "soundfont_id": "sf_a",
        "stem_id": "piano_a",
        "category": "piano",
        "path": str(song_dir),
        "track": 0,
        "out_path": str(sweep_dir / "variants" / "sf_a" / "data" / "0/13/QmTest" / "stem_0.flac"),
    }]).to_csv(sweep_dir / "manifest.csv", index=False)
    variant_dir = sweep_dir / "variants" / "sf_a" / "data" / "0/13/QmTest"
    variant_dir.mkdir(parents=True)
    (variant_dir / "stem_0.flac").write_bytes(b"x")

    probe_path = tmp_path / "probe_stems.yaml"
    probe_path.write_text(yaml.dump({
        "stems": [
            {"id": "piano_a", "category": "piano", "song_id": "0/13/QmTest", "track": 0},
        ],
    }))

    catalog = SweepCatalog("patch", sweep_dir, tmp_path / "basic", probe_stems_path=probe_path)
    cards = build_patch_verify_swipe_cards(catalog, {"piano": ["sf_a", "sf_missing"]})
    assert len(cards) == 1
    assert cards[0]["variant_id"] == "sf_a"
    assert cards[0]["category"] == "piano"


def test_build_patch_verify_swipe_cards_prefers_organ_2(tmp_path: Path):
    from experiments.listening.catalog import SweepCatalog
    from experiments.listening.verification import build_patch_verify_swipe_cards
    import yaml

    sweep_dir = tmp_path / "phase1_archive_soundfonts"
    song_a = tmp_path / "basic" / "data" / "0/13/QmOrganA"
    song_b = tmp_path / "basic" / "data" / "0/13/QmOrganB"
    for song_dir in (song_a, song_b):
        song_dir.mkdir(parents=True)
        (song_dir / "stem_0.flac").write_bytes(b"x")

    clip_rows = []
    for stem_id, song_dir in (("organ", song_a), ("organ_2", song_b)):
        clip_path = (
            sweep_dir / "clips" / "variants" / "sf_a" / "data" / song_dir.name / "stem_0_c0.mp3"
        )
        # unique path under song_id
        clip_path = (
            sweep_dir
            / "clips"
            / "variants"
            / "sf_a"
            / "data"
            / "0/13"
            / song_dir.name
            / "stem_0_c0.mp3"
        )
        clip_path.parent.mkdir(parents=True, exist_ok=True)
        clip_path.write_bytes(b"mp3")
        clip_rows.append({
            "phase": "phase1_archive_soundfonts",
            "variant_id": "sf_a",
            "soundfont_id": "sf_a",
            "stem_id": stem_id,
            "category": "organ",
            "path": str(song_dir),
            "track": 0,
            "out_path": str(clip_path),
            "clip_id": f"{stem_id}_c0",
            "clip_index": 0,
        })
        variant_dir = sweep_dir / "variants" / "sf_a" / "data" / "0/13" / song_dir.name
        variant_dir.mkdir(parents=True, exist_ok=True)
        (variant_dir / "stem_0.flac").write_bytes(b"x")

    pd.DataFrame(clip_rows).to_csv(sweep_dir / "clip_manifest.csv", index=False)
    pd.DataFrame([{
        "variant_id": "sf_a",
        "soundfont_id": "sf_a",
        "stem_id": "organ",
        "category": "organ",
        "path": str(song_a),
        "track": 0,
        "out_path": str(sweep_dir / "variants" / "sf_a" / "data" / "0/13" / song_a.name / "stem_0.flac"),
    }]).to_csv(sweep_dir / "manifest.csv", index=False)

    probe_path = tmp_path / "probe_stems.yaml"
    probe_path.write_text(yaml.dump({
        "stems": [
            {"id": "organ", "category": "organ", "song_id": "0/13/QmOrganA", "track": 0, "note": "organ (GM 22)"},
            {"id": "organ_2", "category": "organ", "song_id": "0/13/QmOrganB", "track": 0, "note": "organ, hymn-style"},
        ],
    }))

    catalog = SweepCatalog("patch", sweep_dir, tmp_path / "basic", probe_stems_path=probe_path)
    cards = build_patch_verify_swipe_cards(catalog, {"organ": ["sf_a"]})
    assert len(cards) == 1
    assert cards[0]["stem_id"] == "organ_2"
    assert "hymn-style" in cards[0]["label"]


def test_build_preset_realify_verification_meta(tmp_path: Path):
    pytest.skip("preset_sweep removed")
