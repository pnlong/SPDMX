"""Copy LICENSE, README, and optional helpers into the released sPDMX dataset directory."""

from __future__ import annotations

import shutil
from pathlib import Path

from shared.config import SPDMX_DATASET_DIR_NAME, SPDMX_DEV_DIR_NAME, SPDMX_FILE_NAME

_TEMPLATE_DIR = Path(__file__).resolve().parent
RELEASE_DOC_NAMES = ("LICENSE", "README.md", "link_single_track_mixes.sh")
LEGACY_TRACK_MAP_NAMES = ("SPDMX.csv", "track_map.csv")


def write_spdmx_release_docs(dataset_dir: str | Path) -> None:
    """Write LICENSE / README / linker script into ``dataset_dir`` (the ``SPDMX/`` folder)."""
    dest = Path(dataset_dir)
    dest.mkdir(parents=True, exist_ok=True)
    for name in RELEASE_DOC_NAMES:
        src = _TEMPLATE_DIR / name
        shutil.copy2(src, dest / name)
        if name.endswith(".sh"):
            dest.joinpath(name).chmod(src.stat().st_mode | 0o111)
    preferred = dest / f"{SPDMX_FILE_NAME}.csv"
    if not preferred.is_file():
        for legacy_name in LEGACY_TRACK_MAP_NAMES:
            legacy = dest / legacy_name
            if legacy.is_file():
                legacy.rename(preferred)
                break


def maybe_write_spdmx_release_docs(dataset_dir: str | Path) -> None:
    """Write release docs into ``SPDMX/`` or ``SPDMX_dev/`` trees."""
    dest = Path(dataset_dir)
    if dest.name not in (SPDMX_DATASET_DIR_NAME, SPDMX_DEV_DIR_NAME):
        return
    write_spdmx_release_docs(dest)
