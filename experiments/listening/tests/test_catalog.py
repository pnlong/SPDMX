"""Tests for sweep listening catalog."""

from pathlib import Path

import pytest
import pandas as pd
import yaml

from experiments.listening.catalog import SweepCatalog, resolve_sweep_catalog_dir


def _write_preset_sweep_tree(tmp_path: Path) -> tuple[Path, Path, Path]:
    source_dir = tmp_path / "basic"
    sweep_dir = tmp_path / "sweep"
    song_id = "0/13/QmTest"
    song_dir = source_dir / "data" / song_id
    song_dir.mkdir(parents=True)
    (song_dir / "stem_0.flac").write_bytes(b"fake")

    variant_dir = sweep_dir / "variants" / "noise0.25_current" / "data" / song_id
    variant_dir.mkdir(parents=True)
    (variant_dir / "stem_0.flac").write_bytes(b"fake")

    manifest = pd.DataFrame([{
        "variant_id": "noise0.25_current",
        "init_noise_level": 0.25,
        "prompt_variant": "current",
        "prompt": "solo piano",
        "stem_id": "piano_test",
        "category": "piano",
        "path": str(song_dir),
        "track": 0,
        "out_path": str(variant_dir / "stem_0.flac"),
    }])
    manifest.to_csv(sweep_dir / "manifest.csv", index=False)

    probe_path = tmp_path / "probe_stems.yaml"
    probe_path.write_text(yaml.dump({
        "stems": [{
            "id": "piano_test",
            "category": "piano",
            "song_id": song_id,
            "track": 0,
            "note": "test piano",
        }],
    }))
    return sweep_dir, source_dir, probe_path


def test_sweep_catalog_preset_removed(tmp_path: Path):
    sweep_dir, source_dir, probe_path = _write_preset_sweep_tree(tmp_path)
    with pytest.raises(RuntimeError, match="preset_sweep"):
        SweepCatalog("preset", sweep_dir, source_dir, probe_stems_path=probe_path)

def _write_patch_sweep_tree(tmp_path: Path) -> tuple[Path, Path, Path]:
    source_dir = tmp_path / "basic"
    sweep_dir = tmp_path / "sweep"
    song_id = "0/13/QmTest"
    song_dir = source_dir / "data" / song_id
    song_dir.mkdir(parents=True)
    (song_dir / "stem_0.flac").write_bytes(b"fake")

    variant_dir = sweep_dir / "variants" / "pool_v1" / "data" / song_id
    variant_dir.mkdir(parents=True)
    (variant_dir / "stem_0.flac").write_bytes(b"fake")

    manifest = pd.DataFrame([{
        "variant_id": "pool_v1",
        "pool_id": "pool_v1",
        "program": 0,
        "gm_class": "piano",
        "stem_id": "piano_test",
        "category": "piano",
        "path": str(song_dir),
        "track": 0,
        "out_path": str(variant_dir / "stem_0.flac"),
    }])
    manifest.to_csv(sweep_dir / "manifest.csv", index=False)

    probe_path = tmp_path / "probe_stems.yaml"
    probe_path.write_text(yaml.dump({
        "stems": [{
            "id": "piano_test",
            "category": "piano",
            "song_id": song_id,
            "track": 0,
        }],
    }))
    return sweep_dir, source_dir, probe_path


def test_sweep_catalog_patch_variants(tmp_path: Path):
    sweep_dir, source_dir, probe_path = _write_patch_sweep_tree(tmp_path)
    catalog = SweepCatalog(
        "patch",
        sweep_dir,
        source_dir,
        probe_stems_path=probe_path,
    )
    variants = catalog.variants()
    assert len(variants) == 1
    assert variants[0]["pool_id"] == "pool_v1"


def test_sweep_catalog_patch_phase2_dedupes_variants_by_id(tmp_path: Path):
    """Phase 2+ manifests repeat variant_id per category soundfont; meta needs blind labels only."""
    source_dir = tmp_path / "basic"
    sweep_dir = tmp_path / "sweep"
    song_a = source_dir / "data" / "0/13/QmA"
    song_b = source_dir / "data" / "0/14/QmB"
    song_a.mkdir(parents=True)
    song_b.mkdir(parents=True)
    (song_a / "stem_0.flac").write_bytes(b"fake")
    (song_b / "stem_0.flac").write_bytes(b"fake")

    rows = []
    for stem_id, category, song_dir, soundfont_id in (
        ("piano_test", "piano", song_a, "airfont_380"),
        ("violin_test", "strings", song_b, "sgm_v2"),
    ):
        for variant_id, fx_profile in (
            ("fx_dry", "dry"),
            ("fx_light", "light"),
            ("fx_warm", "warm"),
        ):
            out_dir = sweep_dir / "variants" / variant_id / "data" / song_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "stem_0.flac").write_bytes(b"fake")
            rows.append({
                "phase": "phase2_fx",
                "variant_id": variant_id,
                "soundfont_id": soundfont_id,
                "fx_profile": fx_profile,
                "pool_id": "",
                "stem_id": stem_id,
                "category": category,
                "path": str(song_dir),
                "track": 0,
                "out_path": str(out_dir / "stem_0.flac"),
            })
    pd.DataFrame(rows).to_csv(sweep_dir / "manifest.csv", index=False)

    probe_path = tmp_path / "probe_stems.yaml"
    probe_path.write_text(yaml.dump({
        "stems": [
            {"id": "piano_test", "category": "piano", "song_id": "0/13/QmA", "track": 0},
            {"id": "violin_test", "category": "strings", "song_id": "0/14/QmB", "track": 0},
        ],
    }))

    catalog = SweepCatalog("patch", sweep_dir, source_dir, probe_stems_path=probe_path)
    variants = catalog.variants()
    assert len(variants) == 3
    assert {v["variant_id"] for v in variants} == {"fx_dry", "fx_light", "fx_warm"}


def test_sweep_catalog_preset_phase2_dedupes_variants_by_id(tmp_path: Path):
    sweep_dir, source_dir, probe_path = _write_preset_sweep_tree(tmp_path)
    with pytest.raises(RuntimeError, match="preset_sweep"):
        SweepCatalog("preset", sweep_dir, source_dir, probe_stems_path=probe_path)

def test_sweep_catalog_patch_phase1_empty_pool_id(tmp_path: Path):
    sweep_dir, source_dir, probe_path = _write_patch_sweep_tree(tmp_path)
    manifest = pd.DataFrame([{
        "phase": "phase1_soundfonts",
        "variant_id": "sgm_v2",
        "soundfont_id": "sgm_v2",
        "soundfont_file": "SGM-V2.01.sf2",
        "fx_profile": "dry",
        "pool_id": "",
        "program": 0,
        "gm_class": "piano",
        "stem_id": "piano_test",
        "category": "piano",
        "path": str(source_dir / "data" / "0/13/QmTest"),
        "track": 0,
        "out_path": str(
            sweep_dir / "variants" / "sgm_v2" / "data" / "0/13/QmTest" / "stem_0.flac"
        ),
    }])
    manifest.to_csv(sweep_dir / "manifest.csv", index=False)

    catalog = SweepCatalog(
        "patch",
        sweep_dir,
        source_dir,
        probe_stems_path=probe_path,
    )
    variants = catalog.variants()
    assert variants[0]["pool_id"] == ""


def test_sweep_catalog_uses_manifest_clip_path_for_reference(tmp_path: Path):
    source_dir = tmp_path / "basic"
    sweep_dir = tmp_path / "phase4"
    clips_dir = sweep_dir / "clips"
    song_id = "0/13/QmClip"
    clip_song_dir = clips_dir / "data" / song_id
    basic_song_dir = source_dir / "data" / song_id
    clip_song_dir.mkdir(parents=True)
    basic_song_dir.mkdir(parents=True)
    (clip_song_dir / "stem_0.mp3").write_bytes(b"ten-second-clip")
    (basic_song_dir / "stem_0.mp3").write_bytes(b"full-length-stem" * 100)

    manifest = pd.DataFrame([{
        "phase": "phase4_verify_diverse",
        "variant_id": "locked",
        "init_noise_level": 0.25,
        "prompt_variant": "current",
        "prompt": "solo piano",
        "stem_id": "piano_clip",
        "category": "piano",
        "path": str(clip_song_dir),
        "track": 0,
        "out_path": str(sweep_dir / "variants" / "locked" / "data" / song_id / "stem_0.mp3"),
    }])
    manifest.to_csv(sweep_dir / "manifest.csv", index=False)
    (sweep_dir / "diverse_stems.yaml").write_text(yaml.dump({
        "clip_seconds": 10,
        "stems": [{
            "id": "piano_clip",
            "category": "piano",
            "song_id": song_id,
            "track": 0,
            "audio_format": "mp3",
        }],
    }))

    sweep_dir, source_dir, probe_path = _write_preset_sweep_tree(tmp_path)
    with pytest.raises(RuntimeError, match="preset_sweep"):
        SweepCatalog("preset", sweep_dir, source_dir, probe_stems_path=probe_path)

def test_resolve_sweep_catalog_dir_prefers_verification_over_root_manifest(
    tmp_path: Path,
    monkeypatch,
):
    output_root = tmp_path / "output"
    phase4 = output_root / "phase4_verify_diverse"
    phase4.mkdir(parents=True)
    (output_root / "manifest.csv").write_text("legacy root manifest\n")
    (phase4 / "manifest.csv").write_text("phase,variant_id\n")

    monkeypatch.setattr(
        "experiments.listening.final_verify.readiness_errors",
        lambda sweep_type, winners_path=None: [],
    )
    monkeypatch.setattr(
        "experiments.listening.final_verify.final_sweep_dir",
        lambda sweep_type, winners_path=None: phase4,
    )

    with pytest.raises(RuntimeError, match="preset_sweep"):
        resolve_sweep_catalog_dir(
            "preset",
            output_root,
            prefer_verification_phase=True,
        )


def test_resolve_sweep_catalog_dir_prefers_phase_with_manifest(tmp_path: Path, monkeypatch):
    output_root = tmp_path / "output"
    phase1 = output_root / "phase1_soundfonts"
    phase2 = output_root / "phase2_fx"
    phase1.mkdir(parents=True)
    phase2.mkdir(parents=True)
    (phase1 / "manifest.csv").write_text("phase,variant_id\n")
    (phase2 / "manifest.csv").write_text("phase,variant_id\n")

    monkeypatch.setattr(
        "experiments.listening.final_verify.readiness_errors",
        lambda sweep_type, winners_path=None: [],
    )
    monkeypatch.setattr(
        "experiments.listening.final_verify.final_sweep_dir",
        lambda sweep_type, winners_path=None: phase2,
    )

    resolved = resolve_sweep_catalog_dir(
        "patch",
        output_root,
        prefer_verification_phase=True,
    )
    assert resolved == phase2.resolve()


def test_resolve_sweep_catalog_dir_honors_explicit_phase_dir(
    tmp_path: Path,
    monkeypatch,
):
    output_root = tmp_path / "output"
    phase2b = output_root / "phase2b_adherence_prompt"
    phase4 = output_root / "phase4_verify_diverse"
    phase2b.mkdir(parents=True)
    phase4.mkdir(parents=True)
    (phase2b / "manifest.csv").write_text("phase,variant_id\n")
    (phase4 / "manifest.csv").write_text("phase,variant_id\n")

    monkeypatch.setattr(
        "experiments.listening.final_verify.readiness_errors",
        lambda sweep_type, winners_path=None: [],
    )
    monkeypatch.setattr(
        "experiments.listening.final_verify.final_sweep_dir",
        lambda sweep_type, winners_path=None: phase4,
    )

    with pytest.raises(RuntimeError, match="preset_sweep"):
        resolve_sweep_catalog_dir(
            "preset",
            phase2b,
            prefer_verification_phase=False,
        )


def test_sweep_catalog_swipe_meta_and_clip_audio(tmp_path: Path):
    source_dir = tmp_path / "basic"
    sweep_dir = tmp_path / "sweep"
    song_id = "0/13/QmTest"
    song_dir = source_dir / "data" / song_id
    song_dir.mkdir(parents=True)
    (song_dir / "stem_0.flac").write_bytes(b"fake")

    clip_path_a = (
        sweep_dir / "clips" / "variants" / "sf_a" / "data" / song_id / "stem_0_c0.mp3"
    )
    clip_path_b = (
        sweep_dir / "clips" / "variants" / "sf_b" / "data" / song_id / "stem_0_c0.mp3"
    )
    clip_path_a.parent.mkdir(parents=True)
    clip_path_b.parent.mkdir(parents=True)
    clip_path_a.write_bytes(b"clip-a")
    clip_path_b.write_bytes(b"clip-b")

    pd.DataFrame([
        {
            "variant_id": "sf_a",
            "stem_id": "piano_test",
            "category": "piano",
            "path": str(song_dir),
            "track": 0,
            "clip_id": "piano_test_c0",
            "clip_index": 0,
            "clip_start_seconds": 10.0,
            "clip_seconds": 10.0,
            "out_path": str(clip_path_a),
        },
        {
            "variant_id": "sf_b",
            "stem_id": "piano_test",
            "category": "piano",
            "path": str(song_dir),
            "track": 0,
            "clip_id": "piano_test_c0",
            "clip_index": 0,
            "clip_start_seconds": 10.0,
            "clip_seconds": 10.0,
            "out_path": str(clip_path_b),
        },
    ]).to_csv(sweep_dir / "clip_manifest.csv", index=False)

    probe_path = tmp_path / "probe_stems.yaml"
    probe_path.write_text(yaml.dump({
        "stems": [{
            "id": "piano_test",
            "category": "piano",
            "song_id": song_id,
            "track": 0,
            "note": "worldmusic piano",
        }],
    }))

    catalog = SweepCatalog(
        "patch",
        sweep_dir,
        source_dir,
        probe_stems_path=probe_path,
    )
    assert catalog.swipe_available() is True

    meta = catalog.get_swipe_meta(category="piano", order="sequential", seed=1)
    assert meta["total_cards"] == 2
    assert meta["mode"] == "swipe"
    assert meta["cards"][0]["audio"]["url"].startswith("/audio/patch/variant/")

    all_meta = catalog.get_swipe_meta(order="sequential", seed=1)
    assert all_meta["total_cards"] == 2
    assert all_meta["category"] is None
    assert set(all_meta["categories"]) == {"piano"}

    resolved = catalog.resolve_variant_audio("sf_a", song_id, "stem_0_c0.mp3")
    assert resolved == clip_path_a.resolve()

