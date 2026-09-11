"""Shared paths and constants for MSS PoC experiments."""

from __future__ import annotations

from pathlib import Path

import yaml

from shared.config import (
    DEV_DIR_NAME,
    EXPERIMENTS_DIR_NAME,
    MUSDB_ROOT as _MUSDB_ROOT,
    OUTPUT_DIR,
    SLAKH_ROOT as _SLAKH_ROOT,
    SPDMX_DATASET_DIR_NAME,
)
from shared.env import path_from_env

REPO_ROOT = Path(__file__).resolve().parents[2]
SEP_DIR = Path(__file__).resolve().parent

# Default on-disk corpora (override via env / config.yaml / shared.config).
# sPDMX PoCs use the **chunked release** tree so external users can reproduce.
SLAKH_ROOT = Path(_SLAKH_ROOT)
SPDMX_ROOT = Path(
    path_from_env(
        "SPDMX_DATASET_ROOT",
        str(Path(OUTPUT_DIR) / SPDMX_DATASET_DIR_NAME),
    )
)
MUSDB_ROOT = Path(_MUSDB_ROOT)

# Experiment outputs under {OUTPUT_DIR}/dev/experiments/separation/
DEV_SEP_DIR = Path(OUTPUT_DIR) / DEV_DIR_NAME / EXPERIMENTS_DIR_NAME / "separation"

TARGETS = ("bass", "drums", "guitar", "piano")
# Train/eval arms: singles + joint (Slakh ∪ SPDMX) under a matched step budget.
MANIFEST_ARMS = ("slakh", "spdmx", "both")
TRAIN_ARMS = MANIFEST_ARMS

DEFAULT_CONFIG_PATH = SEP_DIR / "config.yaml"
DEFAULT_SEED = 43


def load_config(path: Path | None = None) -> dict:
    cfg_path = path or DEFAULT_CONFIG_PATH
    with open(cfg_path) as f:
        return yaml.safe_load(f) or {}


def resolve_dev_dir(cfg: dict | None = None) -> Path:
    cfg = cfg or load_config()
    override = cfg.get("dev_dir")
    if override:
        return Path(override)
    return DEV_SEP_DIR
