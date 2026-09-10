"""Resolve stem files under a chunked sPDMX release (or flat SPDMX_dev)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from shared.config import (
    SPDMX_AUDIO_DIR_NAME,
    SPDMX_MIX_DIR_NAME,
    SPDMX_RAW_DIR_NAME,
    SPDMX_RELEASE_MIX_AUDIO_NAME,
)
from synthesis.chunking import chunk_dir_name, normalize_chunk_id


def resolve_spdmx_stem(
    spdmx_root: Path,
    row: pd.Series,
    *,
    song_id: str,
    track: int,
) -> Path | None:
    """Return an on-disk FLAC path for one stem row, or ``None`` if missing.

    Resolution order (release-first):

    1. ``path`` column (``./chunk_N/<song_id>``) + ``{track}.flac``
    2. ``chunk`` column → ``chunk_N/<song_id>/{track}.flac``
    3. Flat lab trees: ``audio/`` then ``raw/`` under ``spdmx_root``
    """
    root = Path(spdmx_root)
    track_name = f"{int(track)}.flac"

    path_col = row.get("path") if hasattr(row, "get") else None
    if path_col is not None and not (isinstance(path_col, float) and pd.isna(path_col)):
        rel = str(path_col).replace("\\", "/").lstrip("./")
        candidate = root / rel / track_name
        if candidate.is_file():
            return candidate

    if "chunk" in getattr(row, "index", ()) and pd.notna(row.get("chunk")):
        chunk = normalize_chunk_id(row["chunk"])
        candidate = (
            root / chunk_dir_name(chunk) / str(song_id) / track_name
        )
        if candidate.is_file():
            return candidate

    for tree in (SPDMX_AUDIO_DIR_NAME, SPDMX_RAW_DIR_NAME):
        candidate = root / tree / str(song_id) / track_name
        if candidate.is_file():
            return candidate
    return None


def resolve_spdmx_mix(
    spdmx_root: Path,
    row: pd.Series | None = None,
    *,
    song_id: str,
) -> Path | None:
    """Return on-disk full-song mix FLAC, or ``None`` if missing.

    1. ``mix`` column path
    2. ``path`` song dir + ``mix.flac`` (flattened release)
    3. ``chunk`` → ``chunk_N/<song_id>/mix.flac``
    4. Flat ``mix/<song_id>.flac`` under ``spdmx_root``
    """
    root = Path(spdmx_root)
    row = row if row is not None else pd.Series(dtype=object)

    mix_col = row.get("mix") if hasattr(row, "get") else None
    if mix_col is not None and not (isinstance(mix_col, float) and pd.isna(mix_col)):
        rel = str(mix_col).replace("\\", "/").lstrip("./")
        candidate = root / rel
        if candidate.is_file():
            return candidate

    path_col = row.get("path") if hasattr(row, "get") else None
    if path_col is not None and not (isinstance(path_col, float) and pd.isna(path_col)):
        rel = str(path_col).replace("\\", "/").lstrip("./")
        candidate = root / rel / SPDMX_RELEASE_MIX_AUDIO_NAME
        if candidate.is_file():
            return candidate

    if "chunk" in getattr(row, "index", ()) and pd.notna(row.get("chunk")):
        chunk = normalize_chunk_id(row["chunk"])
        candidate = (
            root / chunk_dir_name(chunk) / str(song_id) / SPDMX_RELEASE_MIX_AUDIO_NAME
        )
        if candidate.is_file():
            return candidate

    candidate = root / SPDMX_MIX_DIR_NAME / f"{song_id}.flac"
    if candidate.is_file():
        return candidate
    return None
