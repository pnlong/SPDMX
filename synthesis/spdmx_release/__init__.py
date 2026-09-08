"""Copy LICENSE and README into the released sPDMX dataset directory."""

from __future__ import annotations

import shutil
from pathlib import Path

from shared.config import SPDMX_DATASET_DIR_NAME, SPDMX_DEV_DIR_NAME, SPDMX_FILE_NAME

_TEMPLATE_DIR = Path(__file__).resolve().parent
RELEASE_DOC_NAMES = ("LICENSE", "README.md")
LEGACY_TRACK_MAP_FILE_NAME = "track_map.csv"


def write_spdmx_release_docs(dataset_dir: str | Path) -> None:
    """Write LICENSE and README.md into ``dataset_dir`` (the ``SPDMX/`` folder)."""
    dest = Path(dataset_dir)
    dest.mkdir(parents=True, exist_ok=True)
    for name in RELEASE_DOC_NAMES:
        shutil.copy2(_TEMPLATE_DIR / name, dest / name)
    preferred = dest / f"{SPDMX_FILE_NAME}.csv"
    legacy = dest / LEGACY_TRACK_MAP_FILE_NAME
    if not preferred.is_file() and legacy.is_file():
        legacy.rename(preferred)


def maybe_write_spdmx_release_docs(dataset_dir: str | Path) -> None:
    """Write release docs into ``SPDMX/`` or ``SPDMX_dev/`` trees."""
    dest = Path(dataset_dir)
    if dest.name not in (SPDMX_DATASET_DIR_NAME, SPDMX_DEV_DIR_NAME):
        return
    write_spdmx_release_docs(dest)
