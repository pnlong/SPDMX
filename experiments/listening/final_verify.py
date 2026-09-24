"""Final verification after all tuning phases complete (before lock).

Preset / SA3 paths were removed with ``experiments.preset_sweep``. Patch sweep
verification remains.
"""

from __future__ import annotations

from pathlib import Path

from experiments.listening.catalog import SweepCatalog
from experiments.listening_shared.clips import PRESET_SWEEP_REMOVED
from experiments.patch_sweep.config import (
    EXPERIMENT_DIR as PATCH_EXPERIMENT_DIR,
    PHASE1 as PATCH_PHASE1,
    PHASE1_ARCHIVE as PATCH_PHASE1_ARCHIVE,
    PHASE2 as PATCH_PHASE2,
    PHASE3 as PATCH_PHASE3,
    PHASES as PATCH_PHASES,
    phase_output_dir as patch_phase_output_dir,
)
from experiments.patch_sweep.sweep import default_output_dir as patch_default_output_dir
from experiments.patch_sweep.winners import (
    load_winners as load_patch_winners,
    phase_is_complete as patch_phase_is_complete,
    phase_winners as patch_phase_winners,
)


def _require_patch(sweep_type: str) -> None:
    if sweep_type == "preset":
        raise RuntimeError(PRESET_SWEEP_REMOVED)
    if sweep_type != "patch":
        raise ValueError(f"Unknown sweep type: {sweep_type}")


def _patch_config():
    return {
        "experiment_dir": PATCH_EXPERIMENT_DIR,
        "phases": PATCH_PHASES,
        "required_phases": PATCH_PHASES,
        "phase1": PATCH_PHASE1,
        "phase2": PATCH_PHASE2,
        "phase3": PATCH_PHASE3,
        "load_winners": load_patch_winners,
        "phase_is_complete": patch_phase_is_complete,
        "phase_winners": patch_phase_winners,
        "default_output_dir": patch_default_output_dir,
        "phase_output_dir": patch_phase_output_dir,
    }


def experiment_config(sweep_type: str) -> dict:
    _require_patch(sweep_type)
    return _patch_config()


def winners_path_for(sweep_type: str, path: Path | None = None) -> Path:
    if path is not None:
        return path
    return experiment_config(sweep_type)["experiment_dir"] / "winners.yaml"


def _normalize_winner_ids(value) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, list):
        return {str(item) for item in value if item}
    return {str(value)}


def patch_phase1_sweep_dir(winners_path: Path | None = None) -> Path:
    """Pick phase-1 render output for verification (archive vs legacy 7-bank)."""
    import pandas as pd

    cfg = _patch_config()
    root = cfg["default_output_dir"]()
    legacy_dir = cfg["phase_output_dir"](root, cfg["phase1"])
    archive_dir = cfg["phase_output_dir"](root, PATCH_PHASE1_ARCHIVE)
    path = winners_path_for("patch", winners_path)
    phase1 = cfg["phase_winners"](cfg["phase1"], path)
    winner_ids: set[str] = set()
    for value in phase1.values():
        winner_ids |= _normalize_winner_ids(value)

    def has_manifest(directory: Path) -> bool:
        return (directory / "manifest.csv").is_file()

    if winner_ids and has_manifest(archive_dir):
        archive_ids = set(pd.read_csv(archive_dir / "manifest.csv")["variant_id"].astype(str))
        if winner_ids & archive_ids:
            if not has_manifest(legacy_dir):
                return archive_dir
            legacy_ids = set(pd.read_csv(legacy_dir / "manifest.csv")["variant_id"].astype(str))
            if len(winner_ids & archive_ids) >= len(winner_ids & legacy_ids):
                return archive_dir

    if has_manifest(legacy_dir):
        return legacy_dir
    if has_manifest(archive_dir):
        return archive_dir
    return legacy_dir


def verification_phase(sweep_type: str, winners_path: Path | None = None) -> str:
    del winners_path
    _require_patch(sweep_type)
    return _patch_config()["phase1"]


def readiness_errors(sweep_type: str, winners_path: Path | None = None) -> list[str]:
    _require_patch(sweep_type)
    cfg = _patch_config()
    path = winners_path_for(sweep_type, winners_path)
    errors = []
    for phase in cfg["required_phases"]:
        if not cfg["phase_is_complete"](phase, path):
            errors.append(f"{phase} not complete in {path}")
    sweep_dir = patch_phase1_sweep_dir(path)
    if not (sweep_dir / "manifest.csv").is_file():
        errors.append(f"Missing phase-1 soundfont manifest: {sweep_dir / 'manifest.csv'}")
    return errors


def final_sweep_dir(sweep_type: str, winners_path: Path | None = None) -> Path:
    _require_patch(sweep_type)
    return patch_phase1_sweep_dir(winners_path)


def final_phase_winners(
    sweep_type: str,
    winners_path: Path | None = None,
) -> dict[str, str]:
    """Per-category variant ids from the verification phase (locked in winners.yaml)."""
    cfg = experiment_config(sweep_type)
    path = winners_path_for(sweep_type, winners_path)
    phase = verification_phase(sweep_type, path)
    return dict(cfg["phase_winners"](phase, path))


def final_catalog(
    sweep_type: str,
    winners_path: Path | None = None,
) -> tuple[SweepCatalog, str]:
    errors = readiness_errors(sweep_type, winners_path)
    if errors:
        raise RuntimeError("; ".join(errors))
    phase = verification_phase(sweep_type, winners_path)
    catalog = SweepCatalog(sweep_type, final_sweep_dir(sweep_type, winners_path))
    return catalog, phase


def composed_config(
    sweep_type: str,
    category: str,
    variant_id: str,
    winners_path: Path | None = None,
) -> dict:
    """Full per-category production config for a final-phase variant."""
    del category, winners_path
    _require_patch(sweep_type)
    return {
        "variant_id": variant_id,
        "soundfont_id": variant_id,
        "fx_profile": "dry",
    }


def apply_verification_to_winners(
    verification: dict,
    *,
    sweep_type: str,
    winners_path: Path | None = None,
) -> dict:
    """Override final-phase winners in winners.yaml from verification JSON."""
    _require_patch(sweep_type)
    from experiments.listening.verification import winners_from_verification
    from experiments.patch_sweep.winners import record_phase_winners as record_patch

    winner_map = winners_from_verification(verification, sweep_type=sweep_type)
    if not winner_map:
        raise RuntimeError("No winners in verification file.")

    path = winners_path_for(sweep_type, winners_path)
    return record_patch(PATCH_PHASE1, winner_map, path=path)
