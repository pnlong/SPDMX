"""Post-render song→chunk assignment and packaged CSV helpers.

Production render stays flat (``audio/``, ``mid/``). These helpers support the
``build_spdmx`` packaging step that lays out ``chunk_N/`` trees.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from shared.config import SPDMX_AUDIO_DIR_NAME, SPDMX_MID_DIR_NAME

# ~25 GiB target per download chunk (on-disk FLAC + MIDI).
CHUNK_BYTES_TARGET = 25 * 1024**3
CHUNK_ASSIGNMENT_SEED = 43
CHUNK_DIR_PREFIX = "chunk_"
CHUNKS_FILE_NAME = "chunks.csv"
SHA256SUMS_FILE_NAME = "SHA256SUMS"

FLAT_TRACK_MAP_COLUMNS = [
    "song_id",
    "path",
    "mid",
    "track",
    "original_track",
    "program",
    "is_drum",
    "name",
]

PACKAGED_TRACK_MAP_COLUMNS = FLAT_TRACK_MAP_COLUMNS + ["chunk"]

CHUNKS_CSV_COLUMNS = [
    "chunk",
    "n_songs",
    "n_stems",
    "bytes",
    "archive",
    "sha256",
]


def format_chunk_id(index: int) -> str:
    if index < 0:
        raise ValueError(f"chunk index must be >= 0, got {index}")
    return str(int(index))


def chunk_dir_name(chunk_id: str | int) -> str:
    return f"{CHUNK_DIR_PREFIX}{normalize_chunk_id(chunk_id)}"


def normalize_chunk_id(chunk_id: str | int) -> str:
    """Canonical chunk id: ``\"0\"``, ``\"1\"``, … (no zero-padding)."""
    text = str(chunk_id).strip()
    if text.startswith(CHUNK_DIR_PREFIX):
        text = text[len(CHUNK_DIR_PREFIX) :]
    return str(int(text))


def parse_chunk_id(chunk_dir: str) -> str:
    name = Path(chunk_dir).name
    if not name.startswith(CHUNK_DIR_PREFIX):
        raise ValueError(f"Not a chunk directory name: {chunk_dir}")
    return normalize_chunk_id(name[len(CHUNK_DIR_PREFIX) :])


def packaged_audio_rel(chunk_id: str | int, song_id: str) -> str:
    return f"./{chunk_dir_name(chunk_id)}/{SPDMX_AUDIO_DIR_NAME}/{song_id}"


def packaged_mid_rel(chunk_id: str | int, song_id: str) -> str:
    return f"./{chunk_dir_name(chunk_id)}/{SPDMX_MID_DIR_NAME}/{song_id}.mid"



def directory_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def song_media_bytes(
    song_id: str,
    *,
    audio_root: str | Path,
    mid_root: str | Path,
) -> int:
    audio_dir = Path(audio_root) / song_id
    mid_path = Path(mid_root) / f"{song_id}.mid"
    return directory_size_bytes(audio_dir) + directory_size_bytes(mid_path)


def measure_song_sizes(
    song_ids: Sequence[str],
    *,
    audio_root: str | Path,
    mid_root: str | Path,
) -> dict[str, int]:
    return {
        str(song_id): song_media_bytes(
            str(song_id), audio_root=audio_root, mid_root=mid_root,
        )
        for song_id in song_ids
    }


def assign_songs_to_chunks(
    song_sizes: Mapping[str, int],
    *,
    target_bytes: int = CHUNK_BYTES_TARGET,
    seed: int = CHUNK_ASSIGNMENT_SEED,
) -> dict[str, str]:
    """Greedy-pack songs into ~``target_bytes`` chunks after a seeded shuffle.

    Songs larger than ``target_bytes`` each get their own chunk. Returns
    ``{song_id: chunk_id}`` with unpadded chunk ids (``0``, ``1``, …).
    """
    if target_bytes <= 0:
        raise ValueError(f"target_bytes must be > 0, got {target_bytes}")
    if not song_sizes:
        return {}

    items = sorted(str(s) for s in song_sizes)
    rng = random.Random(seed)
    rng.shuffle(items)

    assignment: dict[str, str] = {}
    chunk_index = 0
    current_bytes = 0

    for song_id in items:
        size = int(song_sizes[song_id])
        if size < 0:
            raise ValueError(f"negative size for song {song_id}: {size}")

        # Start a new chunk when adding would exceed the budget (and the
        # current chunk is non-empty). Oversized songs always open a chunk.
        if current_bytes > 0 and current_bytes + size > target_bytes:
            chunk_index += 1
            current_bytes = 0

        chunk_id = format_chunk_id(chunk_index)
        assignment[song_id] = chunk_id
        current_bytes += size

        if size > target_bytes:
            # Isolate the oversized song; next song starts a fresh chunk.
            chunk_index += 1
            current_bytes = 0

    return assignment


def rewrite_track_map_for_chunks(
    table: pd.DataFrame,
    song_to_chunk: Mapping[str, str],
) -> pd.DataFrame:
    """Return a packaged stems.csv with ``chunk`` and root-relative paths."""
    if "song_id" not in table.columns:
        raise ValueError("stems.csv missing required column: song_id")
    missing = sorted(set(table["song_id"].astype(str)) - set(song_to_chunk))
    if missing:
        preview = ", ".join(missing[:5])
        more = "" if len(missing) <= 5 else f" (+{len(missing) - 5} more)"
        raise ValueError(f"No chunk assignment for song_id(s): {preview}{more}")

    out = table.copy()
    song_ids = out["song_id"].astype(str)
    chunks = song_ids.map(lambda s: normalize_chunk_id(song_to_chunk[s]))
    out["chunk"] = chunks
    out["path"] = [
        packaged_audio_rel(chunk, song)
        for song, chunk in zip(song_ids, chunks, strict=True)
    ]
    out["mid"] = [
        packaged_mid_rel(chunk, song)
        for song, chunk in zip(song_ids, chunks, strict=True)
    ]
    # Stable column order for the packaged product.
    extra = [c for c in out.columns if c not in PACKAGED_TRACK_MAP_COLUMNS]
    return out[PACKAGED_TRACK_MAP_COLUMNS + extra]


def build_chunks_manifest(
    packaged_table: pd.DataFrame,
    song_to_chunk: Mapping[str, str],
    song_sizes: Mapping[str, int],
) -> pd.DataFrame:
    """Build ``chunks.csv`` rows (archive/sha256 empty until distribute)."""
    if packaged_table.empty:
        return pd.DataFrame(columns=CHUNKS_CSV_COLUMNS)

    table = packaged_table.copy()
    table["chunk"] = table["chunk"].map(normalize_chunk_id)
    stems = table.groupby("chunk", sort=False).size()
    songs_per_chunk: dict[str, set[str]] = {}
    for song_id, chunk_id in song_to_chunk.items():
        songs_per_chunk.setdefault(normalize_chunk_id(chunk_id), set()).add(
            str(song_id)
        )

    rows = []
    for chunk_id in sorted(songs_per_chunk, key=int):
        songs = songs_per_chunk[chunk_id]
        rows.append(
            {
                "chunk": chunk_id,
                "n_songs": len(songs),
                "n_stems": int(stems.get(chunk_id, 0)),
                "bytes": int(sum(int(song_sizes.get(s, 0)) for s in songs)),
                "archive": "",
                "sha256": "",
            }
        )
    return pd.DataFrame(rows, columns=CHUNKS_CSV_COLUMNS)


def list_chunk_dirs(dataset_dir: str | Path) -> list[Path]:
    root = Path(dataset_dir)
    dirs = [
        p for p in root.iterdir()
        if p.is_dir() and p.name.startswith(CHUNK_DIR_PREFIX)
    ]
    return sorted(dirs, key=lambda p: int(parse_chunk_id(p.name)))
