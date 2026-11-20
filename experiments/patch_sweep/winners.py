"""Load and update per-phase tuning winners."""

from __future__ import annotations

import hashlib
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from experiments.patch_sweep.config import (
    PHASE1,
    PHASE2,
    PHASES,
    REQUIRED_LOCK_PHASES,
    WINNERS_LOCKED_PATH,
    WINNERS_PATH,
    load_combined_soundfont_catalog,
    load_soundfont_catalog,
    load_yaml,
    soundfont_file_for_id,
)

PhaseWinners = dict[str, str | list[str]]


def _empty_winners_doc() -> dict:
    return {
        "phases": {
            PHASE1: {"completed": False, "winners": {}},
            PHASE2: {"completed": False, "winners": {}},
        }
    }


def load_winners(path: Path = WINNERS_PATH) -> dict:
    if not path.is_file():
        return _empty_winners_doc()
    doc = load_yaml(path)
    if "phases" not in doc:
        doc = _empty_winners_doc()
    for phase in PHASES:
        doc["phases"].setdefault(phase, {"completed": False, "winners": {}})
    return doc


def save_winners(doc: dict, path: Path = WINNERS_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(doc, f, sort_keys=False, default_flow_style=False)
    return path


def record_phase_winners(
    phase: str,
    winners: PhaseWinners,
    *,
    path: Path = WINNERS_PATH,
) -> dict:
    """Persist per-category winners or shortlists for a phase."""
    doc = load_winners(path)
    phase_doc = doc["phases"].setdefault(phase, {"completed": False, "winners": {}})
    phase_doc["winners"] = dict(winners)
    phase_doc["completed"] = True
    phase_doc["recorded_at"] = datetime.now(timezone.utc).isoformat()
    save_winners(doc, path)
    return doc


def phase_winners(phase: str, path: Path = WINNERS_PATH) -> PhaseWinners:
    doc = load_winners(path)
    return dict(doc["phases"].get(phase, {}).get("winners") or {})


def phase_is_complete(phase: str, path: Path = WINNERS_PATH) -> bool:
    doc = load_winners(path)
    return bool(doc["phases"].get(phase, {}).get("completed"))


def _normalize_soundfont_ids(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if v]
    return [str(value)]


def phase1_soundfont_ids(category: str, winners_path: Path = WINNERS_PATH) -> list[str]:
    return _normalize_soundfont_ids(phase_winners(PHASE1, winners_path).get(category))


def resolve_phase1_soundfont_id(category: str, winners_path: Path = WINNERS_PATH) -> str | None:
    """Primary soundfont for a category (first in shortlist)."""
    ids = phase1_soundfont_ids(category, winners_path)
    return ids[0] if ids else None


def _stable_seed_int(*parts: object) -> int:
    """Process-stable RNG seed (unlike ``hash()``, which is randomized per process)."""
    payload = "\0".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFF


def soundfont_seed(sample_seed: int, song_path: str, category: str) -> int:
    """Seed soundfont RNG per (song, listening category)."""
    return _stable_seed_int(sample_seed, song_path, category)


def fx_profile_seed(sample_seed: int, song_path: str, category: str) -> int:
    """Seed FX RNG per (song, listening category), independent of soundfont pick."""
    return _stable_seed_int(sample_seed, song_path, category, "fx")


def pick_soundfont_id(
    soundfont_ids: list[str],
    *,
    category: str,
    song_path: str,
    sample_seed: int,
) -> str:
    if not soundfont_ids:
        raise ValueError(f"No soundfont ids for category: {category}")
    if len(soundfont_ids) == 1:
        return soundfont_ids[0]
    rng = random.Random(soundfont_seed(sample_seed, song_path, category))
    return rng.choice(soundfont_ids)


def pick_fx_profile(
    fx_profiles: list[str],
    *,
    category: str,
    song_path: str,
    sample_seed: int,
) -> str:
    if not fx_profiles:
        raise ValueError(f"No FX profiles for category: {category}")
    if len(fx_profiles) == 1:
        return fx_profiles[0]
    rng = random.Random(fx_profile_seed(sample_seed, song_path, category))
    return rng.choice(fx_profiles)


def fx_profile_from_phase2_variant_id(variant_id: str) -> str:
    """Map phase-2 listening variant ids (fx_dry) to render profiles (dry)."""
    if variant_id in ("dry", "light", "warm", "default"):
        return variant_id
    prefix = "fx_"
    if variant_id.startswith(prefix):
        profile = variant_id[len(prefix):]
        if profile in ("dry", "light", "warm"):
            return profile
    raise ValueError(f"Unknown phase-2 FX variant id: {variant_id!r}")


def phase2_fx_variant_ids(category: str, winners_path: Path = WINNERS_PATH) -> list[str]:
    return _normalize_soundfont_ids(phase_winners(PHASE2, winners_path).get(category))


def resolve_phase2_fx_profile(category: str, winners_path: Path = WINNERS_PATH) -> str | None:
    """Primary FX profile for a category (first in shortlist)."""
    variant_ids = phase2_fx_variant_ids(category, winners_path)
    if not variant_ids:
        return None
    return fx_profile_from_phase2_variant_id(variant_ids[0])


def build_locked_render_config(
    winners_path: Path = WINNERS_PATH,
    soundfonts_catalog_path: Path | None = None,
) -> dict:
    """Merge phase 1–2 winners into production slakh render config."""
    doc = load_winners(winners_path)
    catalog = load_combined_soundfont_catalog() if soundfonts_catalog_path is None else load_soundfont_catalog(soundfonts_catalog_path)

    phase1 = doc["phases"][PHASE1].get("winners") or {}
    phase2 = doc["phases"][PHASE2].get("winners") or {}

    categories = sorted(set(phase1) | set(phase2))
    locked = {"categories": {}}
    for category in categories:
        soundfont_ids = _normalize_soundfont_ids(phase1.get(category))
        if not soundfont_ids:
            continue
        fx_variants = _normalize_soundfont_ids(phase2.get(category, "fx_light"))
        fx_profiles = [
            fx_profile_from_phase2_variant_id(variant_id) for variant_id in fx_variants
        ]
        locked["categories"][category] = {
            "soundfont_ids": soundfont_ids,
            "soundfont_id": soundfont_ids[0],
            "soundfont": soundfont_file_for_id(soundfont_ids[0], catalog),
            "fx_variant_ids": fx_variants,
            "fx_profile": fx_profiles[0],
            "fx_profiles": fx_profiles,
        }

    locked["locked_at"] = datetime.now(timezone.utc).isoformat()
    return locked


def write_locked_config(
    winners_path: Path = WINNERS_PATH,
    output_path: Path = WINNERS_LOCKED_PATH,
) -> Path:
    for phase in REQUIRED_LOCK_PHASES:
        if not phase_is_complete(phase, winners_path):
            raise RuntimeError(f"Phase not complete: {phase}")

    locked = build_locked_render_config(winners_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.safe_dump(locked, f, sort_keys=False, default_flow_style=False)
    return output_path
