"""Paths for lite SAO fine-tune PoC."""

from __future__ import annotations

from pathlib import Path

import yaml

from shared.config import DEV_DIR_NAME, EXPERIMENTS_DIR_NAME, OUTPUT_DIR, SPDMX_DATASET_DIR_NAME, SLAKH_ROOT as _SLAKH_ROOT
from shared.env import path_from_env

SAO_DIR = Path(__file__).resolve().parent
REPO_ROOT = SAO_DIR.parents[1]

SLAKH_ROOT = Path(_SLAKH_ROOT)
SPDMX_ROOT = Path(
    path_from_env(
        "SPDMX_DATASET_ROOT",
        f"{OUTPUT_DIR}/{SPDMX_DATASET_DIR_NAME}",
    )
)

DEV_SAO_DIR = Path(OUTPUT_DIR) / DEV_DIR_NAME / EXPERIMENTS_DIR_NAME / "sao"
DEFAULT_CONFIG_PATH = SAO_DIR / "config.yaml"
ARMS = ("slakh", "spdmx_matched", "spdmx_full")


def load_config(path: Path | None = None) -> dict:
    with open(path or DEFAULT_CONFIG_PATH) as f:
        return yaml.safe_load(f) or {}


def resolve_dev_dir(cfg: dict | None = None) -> Path:
    cfg = cfg or load_config()
    if cfg.get("dev_dir"):
        return Path(cfg["dev_dir"])
    return DEV_SAO_DIR
