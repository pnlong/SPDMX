"""Stable song sharding for multi-machine synthesis."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, Sequence

from shared.config import DATA_DIR_NAME, SPDMX_AUDIO_DIR_NAME, SPDMX_RAW_DIR_NAME


def song_id_from_synthesis_path(path: str | Path) -> str:
    """Extract song id from a synthesis output directory path."""
    parts = Path(path).parts
    for marker in (SPDMX_RAW_DIR_NAME, SPDMX_AUDIO_DIR_NAME, DATA_DIR_NAME):
        if marker in parts:
            idx = parts.index(marker)
            return str(Path(*parts[idx + 1:]))
    raise ValueError(
        f"Song path missing {SPDMX_RAW_DIR_NAME}/, "
        f"{SPDMX_AUDIO_DIR_NAME}/, or {DATA_DIR_NAME}/ segment: {path}"
    )


def validate_shard_args(shard_count: int, shard_index: int) -> None:
    if shard_count < 1:
        raise ValueError(f"--shard-count must be >= 1, got {shard_count}")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(
            f"--shard-index must be in [0, {shard_count - 1}], got {shard_index}"
        )


def partition_slices(n: int, shard_count: int) -> list[tuple[int, int]]:
    """Return half-open ``(start, end)`` bounds for each shard index."""
    base, extra = divmod(n, shard_count)
    slices: list[tuple[int, int]] = []
    start = 0
    for i in range(shard_count):
        size = base + (1 if i < extra else 0)
        slices.append((start, start + size))
        start += size
    return slices


def shuffled_song_ids(song_ids: Sequence[str], shard_count: int) -> list[str]:
    """Sort *song_ids*, shuffle with ``seed=shard_count``, return the list."""
    items = sorted(str(s) for s in song_ids)
    rng = random.Random(shard_count)
    rng.shuffle(items)
    return items


def shard_song_ids(
    song_ids: Sequence[str],
    *,
    shard_count: int,
    shard_index: int,
) -> set[str]:
    """Song ids owned by ``shard_index`` after the stable shuffle."""
    validate_shard_args(shard_count, shard_index)
    if shard_count == 1:
        return {str(s) for s in song_ids}
    shuffled = shuffled_song_ids(song_ids, shard_count)
    start, end = partition_slices(len(shuffled), shard_count)[shard_index]
    return set(shuffled[start:end])


def all_shard_song_ids(
    song_ids: Sequence[str],
    *,
    shard_count: int,
) -> list[set[str]]:
    """Return the song-id set for every shard (for tests)."""
    if shard_count == 1:
        return [{str(s) for s in song_ids}]
    shuffled = shuffled_song_ids(song_ids, shard_count)
    bounds = partition_slices(len(shuffled), shard_count)
    return [set(shuffled[start:end]) for start, end in bounds]


def filter_work_indices_by_shard(
    work_indices: Iterable[int],
    dataset,
    *,
    shard_count: int,
    shard_index: int,
) -> tuple[list[int], int]:
    """Keep indices whose ``song_id`` belongs to this shard.

    Returns ``(filtered_indices, assigned_song_count)`` where assigned count
    is the number of songs in the full dataset assigned to this shard (not
    only the remaining-work subset).
    """
    all_song_ids = dataset["song_id"].astype(str).tolist()
    assigned = shard_song_ids(
        all_song_ids, shard_count=shard_count, shard_index=shard_index,
    )
    assigned_count = len(assigned)
    if shard_count == 1:
        return list(work_indices), assigned_count
    filtered = [
        i for i in work_indices if str(dataset.at[i, "song_id"]) in assigned
    ]
    return filtered, assigned_count


def format_shard_summary(
    shard_count: int,
    shard_index: int,
    assigned: int,
    remaining: int,
) -> str:
    if shard_count <= 1:
        return f"Processing {remaining} songs."
    return (
        f"Shard {shard_index + 1}/{shard_count}: "
        f"{assigned} songs assigned, {remaining} still to render."
    )
