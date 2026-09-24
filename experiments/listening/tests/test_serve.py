"""Tests for sweep listening HTTP server."""

import json
from http import HTTPStatus
from io import BytesIO
from pathlib import Path

import pandas as pd
import pytest
import yaml

from experiments.listening.catalog import SweepCatalog
from experiments.listening.serve import (
    STATIC_DIR,
    SweepListeningHandler,
    parse_sweep_audio,
)


def _write_sweep(tmp_path: Path) -> SweepCatalog:
    sweep_dir = tmp_path / "sweep"
    source_dir = tmp_path / "basic"
    song_id = "0/13/QmTest"
    song_dir = source_dir / "data" / song_id
    song_dir.mkdir(parents=True)
    (song_dir / "stem_0.flac").write_bytes(b"fake")

    variant_dir = sweep_dir / "variants" / "sgm" / "data" / song_id
    variant_dir.mkdir(parents=True)
    (variant_dir / "stem_0.flac").write_bytes(b"fake")

    pd.DataFrame([{
        "variant_id": "sgm",
        "soundfont_id": "sgm",
        "fx_profile": "dry",
        "phase": "phase1_soundfonts",
        "stem_id": "piano_test",
        "category": "piano",
        "path": str(song_dir),
        "track": 0,
        "out_path": str(variant_dir / "stem_0.flac"),
    }]).to_csv(sweep_dir / "manifest.csv", index=False)

    probe_path = tmp_path / "probe_stems.yaml"
    probe_path.write_text(yaml.dump({
        "stems": [{
            "id": "piano_test",
            "category": "piano",
            "song_id": song_id,
            "track": 0,
        }],
    }))
    return SweepCatalog("patch", sweep_dir, source_dir, probe_stems_path=probe_path)


def _handler(catalog: SweepCatalog) -> SweepListeningHandler:
    SweepListeningHandler.catalogs = {"patch": catalog}
    SweepListeningHandler.default_sweep = "patch"
    SweepListeningHandler.static_dir = STATIC_DIR.resolve()
    handler = SweepListeningHandler.__new__(SweepListeningHandler)
    handler._last_status = 0
    handler.request_version = "HTTP/1.1"
    handler.headers = {"Content-Length": "0"}
    handler.rfile = BytesIO(b"{}")
    handler.wfile = BytesIO()

    def send_response(code, message=None):
        handler._last_status = code

    handler.send_response = send_response
    handler.send_header = lambda *args, **kwargs: None
    handler.end_headers = lambda: None
    return handler


def test_serves_test_page(tmp_path: Path):
    catalog = _write_sweep(tmp_path)
    handler = _handler(catalog)
    handler.path = "/test"
    handler.do_GET()
    assert handler._last_status == HTTPStatus.OK
    assert b"Rate each sample" in handler.wfile.getvalue()


def test_api_meta(tmp_path: Path):
    catalog = _write_sweep(tmp_path)
    handler = _handler(catalog)
    handler.path = "/api/patch/meta?session_seed=42"
    handler.do_GET()
    assert handler._last_status == HTTPStatus.OK
    payload = __import__("json").loads(handler.wfile.getvalue())
    assert payload["sweep_type"] == "patch"
    assert len(payload["stems"]) == 1


def _write_patch_sweep(tmp_path: Path) -> SweepCatalog:
    sweep_dir = tmp_path / "sweep"
    source_dir = tmp_path / "basic"
    song_id = "0/13/QmTest"
    song_dir = source_dir / "data" / song_id
    song_dir.mkdir(parents=True)
    (song_dir / "stem_0.flac").write_bytes(b"fake")

    variant_dir = sweep_dir / "variants" / "sgm_v2" / "data" / song_id
    variant_dir.mkdir(parents=True)
    (variant_dir / "stem_0.flac").write_bytes(b"fake")

    pd.DataFrame([{
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
        "path": str(song_dir),
        "track": 0,
        "out_path": str(variant_dir / "stem_0.flac"),
    }]).to_csv(sweep_dir / "manifest.csv", index=False)

    probe_path = tmp_path / "probe_stems.yaml"
    probe_path.write_text(yaml.dump({
        "stems": [{
            "id": "piano_test",
            "category": "piano",
            "song_id": song_id,
            "track": 0,
        }],
    }))
    return SweepCatalog("patch", sweep_dir, source_dir, probe_stems_path=probe_path)


def test_api_meta_patch_phase1_json(tmp_path: Path):
    catalog = _write_patch_sweep(tmp_path)
    handler = _handler(catalog)
    handler.catalogs = {"patch": catalog}
    handler.path = "/api/patch/meta?session_seed=42"
    handler.do_GET()
    assert handler._last_status == HTTPStatus.OK
    body = handler.wfile.getvalue()
    assert b": NaN" not in body
    payload = __import__("json").loads(body)
    assert payload["sweep_type"] == "patch"
    assert payload["variants"][0]["pool_id"] in ("", None)


def test_parse_sweep_audio_paths():
    ref = parse_sweep_audio("/audio/patch/reference/piano_test/stem_0.flac")
    assert ref == ("patch", "reference", "piano_test", "stem_0.flac")

    var = parse_sweep_audio("/audio/patch/variant/sgm/0/13/QmTest/stem_0.flac")
    assert var == ("patch", "variant", "sgm|0/13/QmTest", "stem_0.flac")


def test_api_responses_session_and_checkpoint(tmp_path: Path):
    catalog = _write_sweep(tmp_path)
    handler = _handler(catalog)

    handler.path = "/api/patch/responses/session"
    handler.do_GET()
    assert handler._last_status == HTTPStatus.OK
    payload = __import__("json").loads(handler.wfile.getvalue())
    assert payload["ratings"] == []

    body = __import__("json").dumps({
        "checkpoint": True,
        "ratings": [{
            "stem_id": "piano_test",
            "category": "piano",
            "samples": [{
                "variant_id": "sgm",
                "blind_label": "A",
                "content": 4,
                "realism": 5,
            }],
        }],
    }).encode("utf-8")
    handler.path = "/api/patch/responses"
    handler.headers = {"Content-Length": str(len(body))}
    handler.rfile = BytesIO(body)
    handler.wfile = BytesIO()
    handler.do_POST()
    assert handler._last_status == HTTPStatus.OK
    result = __import__("json").loads(handler.wfile.getvalue())
    assert result["checkpoint"] is True
    session_path = catalog.session_responses_path()
    assert session_path.is_file()
    saved = __import__("json").loads(session_path.read_text())
    assert saved["ratings"][0]["stem_id"] == "piano_test"
    assert "checkpoint" not in saved

    handler.path = "/api/patch/responses/session"
    handler.wfile = BytesIO()
    handler.do_GET()
    reloaded = __import__("json").loads(handler.wfile.getvalue())
    assert reloaded["ratings"][0]["stem_id"] == "piano_test"


def test_verify_meta_and_save(tmp_path: Path, monkeypatch):
    pytest.skip("preset verify meta path removed with preset_sweep; patch verify needs rewrite")
    catalog = _write_sweep(tmp_path)
    responses_dir = catalog.responses_dir()
    responses_path = responses_dir / "responses_test.json"
    responses_path.write_text(json.dumps({
        "ratings": [{
            "stem_id": "piano_test",
            "category": "piano",
            "samples": [
                {"variant_id": "sgm", "content": 5, "realism": 4},
            ],
        }],
    }))

    monkeypatch.setattr(
        "experiments.listening.final_verify.final_catalog",
        lambda sweep_type, winners_path=None: (catalog, "phase2_prompts"),
    )
    monkeypatch.setattr(
        "experiments.listening.final_verify.readiness_errors",
        lambda sweep_type, winners_path=None: [],
    )
    monkeypatch.setattr(
        "experiments.listening.final_verify.composed_config",
        lambda sweep_type, category, variant_id, winners_path=None: {
            "variant_id": variant_id,
            "init_noise_level": 0.25,
            "prompt_variant": "current",
        },
    )

    handler = _handler(catalog)
    handler.path = "/api/patch/verify/meta?responses=responses_test.json"
    handler.do_GET()
    assert handler._last_status == HTTPStatus.OK
    meta = json.loads(handler.wfile.getvalue())
    assert meta["source_responses"] == "responses_test.json"
    assert meta["mode"] == "final"
    assert meta["categories"][0]["category"] == "piano"

    body = json.dumps({
        "mode": "verification",
        "source_responses": "responses_test.json",
        "categories": [{
            "category": "piano",
            "approved": ["sgm"],
            "winner_variant_id": "sgm",
        }],
    }).encode("utf-8")
    handler.path = "/api/patch/responses"
    handler.headers = {"Content-Length": str(len(body))}
    handler.rfile = BytesIO(body)
    handler.wfile = BytesIO()
    handler.do_POST()
    assert handler._last_status == HTTPStatus.OK
    saved_files = list(responses_dir.glob("verification_final_responses_test_*.json"))
    assert saved_files



@pytest.mark.skip(reason="preset_sweep removed")
def test_verify_preset_meta_from_winners_yaml(tmp_path: Path, monkeypatch):
    catalog = _write_sweep(tmp_path)

    monkeypatch.setattr(
        "experiments.listening.final_verify.final_catalog",
        lambda sweep_type, winners_path=None: (catalog, "phase3_diffusion"),
    )
    monkeypatch.setattr(
        "experiments.listening.final_verify.readiness_errors",
        lambda sweep_type, winners_path=None: [],
    )
    monkeypatch.setattr(
        "experiments.listening.final_verify.final_phase_winners",
        lambda sweep_type, winners_path=None: {"piano": "sgm"},
    )
    monkeypatch.setattr(
        "experiments.listening.final_verify.composed_config",
        lambda sweep_type, category, variant_id, winners_path=None: {
            "variant_id": variant_id,
            "init_noise_level": 0.25,
            "prompt_variant": "current",
            "steps": 8,
            "cfg_scale": 1.0,
        },
    )

    handler = _handler(catalog)
    handler.path = "/api/patch/verify/meta"
    handler.do_GET()
    assert handler._last_status == HTTPStatus.OK
    meta = json.loads(handler.wfile.getvalue())
    assert meta["verification_mode"] == "patch_verify"
    assert meta["source_responses"] == "winners.yaml"
    assert meta["categories"][0]["category"] == "piano"


def test_serve_audio_falls_back_to_verification_catalog(tmp_path: Path, monkeypatch):
    output_root = tmp_path / "output"
    phase_dir = output_root / "phase2_fx"
    source_dir = tmp_path / "basic"
    song_id = "0/38/QmBand"
    song_dir = source_dir / "data" / song_id
    song_dir.mkdir(parents=True)
    (song_dir / "stem_12.mp3").write_bytes(b"fake-ref")

    variant_dir = phase_dir / "variants" / "fx_dry" / "data" / song_id
    variant_dir.mkdir(parents=True)
    (variant_dir / "stem_12.mp3").write_bytes(b"fake-variant")

    pd.DataFrame([{
        "phase": "phase2_fx",
        "variant_id": "fx_dry",
        "soundfont_id": "arachno",
        "soundfont_file": "Arachno.sf2",
        "fx_profile": "dry",
        "pool_id": "",
        "program": 0,
        "gm_class": "drums",
        "stem_id": "drums_band",
        "category": "drums",
        "path": str(song_dir),
        "track": 12,
        "out_path": str(variant_dir / "stem_12.mp3"),
    }]).to_csv(phase_dir / "manifest.csv", index=False)

    probe_path = tmp_path / "probe_stems.yaml"
    probe_path.write_text(yaml.dump({
        "stems": [{
            "id": "drums_band",
            "category": "drums",
            "song_id": song_id,
            "track": 12,
        }],
    }))

    root_catalog = SweepCatalog("patch", output_root, source_dir, probe_stems_path=probe_path)
    verify_catalog = SweepCatalog("patch", phase_dir, source_dir, probe_stems_path=probe_path)

    handler = _handler(root_catalog)
    handler.catalogs = {"patch": root_catalog}
    monkeypatch.setattr(
        "experiments.listening.final_verify.readiness_errors",
        lambda sweep_type, winners_path=None: [],
    )
    monkeypatch.setattr(
        "experiments.listening.final_verify.final_catalog",
        lambda sweep_type, winners_path=None: (verify_catalog, "phase2_fx"),
    )

    handler.path = (
        "/audio/patch/variant/fx_dry/0/38/QmBand/stem_12.mp3"
    )
    handler.wfile = BytesIO()
    handler.do_GET()
    assert handler._last_status == HTTPStatus.OK
    assert handler.wfile.getvalue() == b"fake-variant"


def _write_patch_swipe_sweep(tmp_path: Path) -> SweepCatalog:
    sweep_dir = tmp_path / "sweep"
    source_dir = tmp_path / "basic"
    song_id = "0/13/QmTest"
    song_dir = source_dir / "data" / song_id
    song_dir.mkdir(parents=True)
    (song_dir / "stem_0.flac").write_bytes(b"fake")

    clip_path = (
        sweep_dir / "clips" / "variants" / "sf_a" / "data" / song_id / "stem_0_c0.mp3"
    )
    clip_path.parent.mkdir(parents=True)
    clip_path.write_bytes(b"clip")

    pd.DataFrame([{
        "variant_id": "sf_a",
        "stem_id": "piano_test",
        "category": "piano",
        "path": str(song_dir),
        "track": 0,
        "clip_id": "piano_test_c0",
        "clip_index": 0,
        "clip_start_seconds": 10.0,
        "clip_seconds": 10.0,
        "out_path": str(clip_path),
    }]).to_csv(sweep_dir / "clip_manifest.csv", index=False)

    probe_path = tmp_path / "probe_stems.yaml"
    probe_path.write_text(yaml.dump({
        "stems": [{
            "id": "piano_test",
            "category": "piano",
            "song_id": song_id,
            "track": 0,
        }],
    }))
    return SweepCatalog("patch", sweep_dir, source_dir, probe_stems_path=probe_path)


def test_api_swipe_meta_and_checkpoint(tmp_path: Path):
    catalog = _write_patch_swipe_sweep(tmp_path)
    handler = _handler(catalog)
    handler.catalogs = {"patch": catalog}

    handler.path = "/api/patch/swipe/meta?category=piano&seed=7&order=sequential"
    handler.do_GET()
    assert handler._last_status == HTTPStatus.OK
    payload = json.loads(handler.wfile.getvalue())
    assert payload["mode"] == "swipe"
    assert payload["total_cards"] == 1
    assert payload["cards"][0]["variant_id"] == "sf_a"

    handler.path = "/api/patch/swipe/responses/session"
    handler.wfile = BytesIO()
    handler.do_GET()
    assert handler._last_status == HTTPStatus.OK
    assert json.loads(handler.wfile.getvalue())["votes"] == []

    body = json.dumps({
        "mode": "swipe",
        "checkpoint": True,
        "votes": [{
            "category": "piano",
            "stem_id": "piano_test",
            "clip_id": "piano_test_c0",
            "variant_id": "sf_a",
            "tier": "strong_accept",
        }],
    }).encode("utf-8")
    handler.path = "/api/patch/responses"
    handler.headers = {"Content-Length": str(len(body))}
    handler.rfile = BytesIO(body)
    handler.wfile = BytesIO()
    handler.do_POST()
    assert handler._last_status == HTTPStatus.OK

    session_path = catalog.swipe_session_responses_path()
    assert session_path.is_file()
    saved = json.loads(session_path.read_text())
    assert saved["votes"][0]["tier"] == "strong_accept"

    handler.path = "/swipe"
    handler.wfile = BytesIO()
    handler.do_GET()
    assert handler._last_status == HTTPStatus.OK
    assert b"Swipe" in handler.wfile.getvalue() or b"swipe" in handler.wfile.getvalue().lower()

