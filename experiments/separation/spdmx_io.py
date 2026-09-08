"""Resolve stem files under a chunked sPDMX release (or flat SPDMX_dev fallback)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from shared.config import SPDMX_AUDIO_DIR_NAME, SPDMX_RAW_DIR_NAME
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

    1. ``path`` column (``./chunk_N/audio/<song_id>``) + ``{track}.flac``
    2. ``chunk`` column → ``chunk_N/audio/<song_id>/{track}.flac``
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
            root / chunk_dir_name(chunk) / SPDMX_AUDIO_DIR_NAME / str(song_id) / track_name
        )
        if candidate.is_file():
            return candidate

    for tree in (SPDMX_AUDIO_DIR_NAME, SPDMX_RAW_DIR_NAME):
        candidate = root / tree / str(song_id) / track_name
        if candidate.is_file():
            return candidate
    return None
