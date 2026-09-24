"""Output path helpers for experiments."""

from __future__ import annotations

from pathlib import Path

from shared.config import (
    DEV_DIR_NAME,
    EXPERIMENTS_DIR_NAME,
    PATCH_SWEEP_DIR_NAME,
)
from synthesis.paths import dev_root

EXPERIMENTS_DIR = Path(__file__).resolve().parent
DEFAULT_PROBE_STEMS = EXPERIMENTS_DIR / "probe_stems.yaml"


def experiments_root(output_dir: str) -> str:
    return f"{dev_root(output_dir)}/{EXPERIMENTS_DIR_NAME}"


def patch_sweep_output_root(output_dir: str) -> str:
    return f"{experiments_root(output_dir)}/{PATCH_SWEEP_DIR_NAME}"
