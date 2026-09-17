"""Paths and config for multi-instrument transcription (C1) pilots."""

from __future__ import annotations

from pathlib import Path

import yaml

from shared.config import (
    DEV_DIR_NAME,
    EXPERIMENTS_DIR_NAME,
    OUTPUT_DIR,
    SLAKH_ROOT as _SLAKH_ROOT,
    SPDMX_DATASET_DIR_NAME,
)
from shared.env import path_from_env

REPO_ROOT = Path(__file__).resolve().parents[2]
TRANS_DIR = Path(__file__).resolve().parent

SLAKH_ROOT = Path(_SLAKH_ROOT)
SPDMX_ROOT = Path(
    path_from_env(
        "SPDMX_DATASET_ROOT",
        str(Path(OUTPUT_DIR) / SPDMX_DATASET_DIR_NAME),
    )
)

DEV_TRANS_DIR = Path(OUTPUT_DIR) / DEV_DIR_NAME / EXPERIMENTS_DIR_NAME / "transcription"
DEFAULT_CONFIG_PATH = TRANS_DIR / "config.yaml"
ARMS = ("slakh", "spdmx", "both")


def load_config(path: Path | None = None) -> dict:
    cfg_path = path or DEFAULT_CONFIG_PATH
    with open(cfg_path) as f:
        return yaml.safe_load(f) or {}


def resolve_dev_dir(cfg: dict | None = None) -> Path:
    cfg = cfg or load_config()
    override = cfg.get("dev_dir")
    if override:
        return Path(override)
    return DEV_TRANS_DIR
