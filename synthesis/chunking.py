"""Post-render song→chunk assignment and packaged CSV helpers.

Production render stays flat (``audio/``, ``mid/``, ``mix/``). These helpers
support the ``build_spdmx`` packaging step that lays out flattened
``chunk_N/<song_id>/`` trees (stems + ``mix.flac`` + ``mix.mid``).
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from shared.config import (
    SPDMX_MIX_DIR_NAME,
    SPDMX_MULTITRACK_SUBSET_COLUMN,
    SPDMX_RELEASE_MIX_AUDIO_NAME,
    SPDMX_RELEASE_MIX_MIDI_NAME,
)

# Fixed number of roughly equal-sized download chunks (LPT bin packing).
DEFAULT_NUM_CHUNKS = 64
# Legacy soft budget (plots / docs may still mention ~GiB scale).
CHUNK_BYTES_TARGET = 25 * 1024**3
CHUNK_ASSIGNMENT_SEED = 43
CHUNK_DIR_PREFIX = "chunk_"
CHUNKS_FILE_NAME = "chunks.csv"
CHUNK_ASSIGNMENT_FILE_NAME = "chunk_assignment.csv"
PARTS_FILE_NAME = "PARTS.md"
SHA256SUMS_FILE_NAME = "SHA256SUMS"
# Mozilla Data Collective publishes two archives split by chunk-id parity.
DEFAULT_NUM_MDC_PARTS = 2
# Stay under the 1 TB (10^12 byte) upload cap with room for tar/gzip overhead.
DEFAULT_MAX_PART_BYTES = 950 * 10**9
CHUNK_ASSIGNMENT_COLUMNS = ["song_id", "chunk"]

FLAT_TRACK_MAP_COLUMNS = [
    "song_id",
    "path",
    "mid",
    "mix",
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


def packaged_song_rel(chunk_id: str | int, song_id: str) -> str:
    """Release-relative song directory: ``./chunk_N/<song_id>``."""
    return f"./{chunk_dir_name(chunk_id)}/{song_id}"


def packaged_audio_rel(chunk_id: str | int, song_id: str) -> str:
    """Stem directory (same as song dir in the flattened layout)."""
    return packaged_song_rel(chunk_id, song_id)


def packaged_mid_rel(chunk_id: str | int, song_id: str) -> str:
    return f"{packaged_song_rel(chunk_id, song_id)}/{SPDMX_RELEASE_MIX_MIDI_NAME}"


def packaged_mix_rel(chunk_id: str | int, song_id: str) -> str:
    return f"{packaged_song_rel(chunk_id, song_id)}/{SPDMX_RELEASE_MIX_AUDIO_NAME}"


def flat_mix_rel(song_id: str) -> str:
    return f"./{SPDMX_MIX_DIR_NAME}/{song_id}.flac"


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
    mix_root: str | Path | None = None,
    include_mix: bool = True,
) -> int:
    audio_dir = Path(audio_root) / song_id
    mid_path = Path(mid_root) / f"{song_id}.mid"
    total = directory_size_bytes(audio_dir) + directory_size_bytes(mid_path)
    if include_mix and mix_root is not None:
        total += directory_size_bytes(Path(mix_root) / f"{song_id}.flac")
    return total


def measure_song_sizes(
    song_ids: Sequence[str],
    *,
    audio_root: str | Path,
    mid_root: str | Path,
    mix_root: str | Path | None = None,
    include_mix_by_song: Mapping[str, bool] | None = None,
) -> dict[str, int]:
    return {
        str(song_id): song_media_bytes(
            str(song_id),
            audio_root=audio_root,
            mid_root=mid_root,
            mix_root=mix_root,
            include_mix=(
                True
                if include_mix_by_song is None
                else bool(include_mix_by_song.get(str(song_id), True))
            ),
        )
        for song_id in song_ids
    }


def bundled_chunk_assignment_path() -> Path:
    """Checked-in v1 ``song_id,chunk`` map (historical; not trimmed on rebuild)."""
    return (
        Path(__file__).resolve().parent
        / "spdmx_release"
        / CHUNK_ASSIGNMENT_FILE_NAME
    )


def load_chunk_assignment(path: str | Path) -> dict[str, str]:
    """Load ``{song_id: chunk_id}`` from a ``song_id,chunk`` CSV."""
    frame = pd.read_csv(path, usecols=CHUNK_ASSIGNMENT_COLUMNS)
    if frame.empty:
        return {}
    song_ids = frame["song_id"].astype(str)
    duplicated = song_ids.duplicated()
    if bool(duplicated.any()):
        preview = ", ".join(song_ids[duplicated].head(5))
        raise ValueError(f"duplicate song_id in {path}: {preview}")
    return {
        str(song_id): normalize_chunk_id(chunk)
        for song_id, chunk in zip(frame["song_id"], frame["chunk"], strict=True)
    }


def merge_chunk_assignment(
    frozen: Mapping[str, str] | None,
    assignment: Mapping[str, str],
) -> pd.DataFrame:
    """Historical frozen rows plus songs assigned for the first time.

    Frozen chunk ids win. Songs listed only in ``frozen`` stay in the table
    even when this render dropped them.
    """
    rows: dict[str, str] = {}
    if frozen:
        for song_id, chunk_id in frozen.items():
            rows[str(song_id)] = normalize_chunk_id(chunk_id)
    for song_id, chunk_id in assignment.items():
        key = str(song_id)
        if key not in rows:
            rows[key] = normalize_chunk_id(chunk_id)
    ordered = sorted(rows)
    return pd.DataFrame(
        {
            "song_id": ordered,
            "chunk": [int(rows[song_id]) for song_id in ordered],
        },
        columns=CHUNK_ASSIGNMENT_COLUMNS,
    )


def write_chunk_assignment(
    path: str | Path,
    frozen: Mapping[str, str] | None,
    assignment: Mapping[str, str],
) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    merge_chunk_assignment(frozen, assignment).to_csv(
        dest, index=False, lineterminator="\n"
    )
    return dest


def mdc_part_for_chunk(chunk_id: str | int, *, num_parts: int = DEFAULT_NUM_MDC_PARTS) -> int:
    """Part number (1-based) for a chunk. Two parts use chunk-id parity.

    Even ids (``0, 2, …``) are part 1. Odd ids are part 2. The rule does not
    look at file sizes, so a song that keeps its chunk stays in the same part.
    """
    if int(num_parts) != DEFAULT_NUM_MDC_PARTS:
        raise ValueError(
            f"MDC part split supports {DEFAULT_NUM_MDC_PARTS} parts, got {num_parts}"
        )
    return (int(normalize_chunk_id(chunk_id)) % 2) + 1


def partition_chunk_ids(
    chunk_ids: Sequence[str | int],
    *,
    num_parts: int = DEFAULT_NUM_MDC_PARTS,
) -> dict[int, list[str]]:
    """Group chunk ids into MDC parts. Keys are 1 .. ``num_parts``."""
    parts: dict[int, list[str]] = {part: [] for part in range(1, int(num_parts) + 1)}
    for chunk_id in chunk_ids:
        part = mdc_part_for_chunk(chunk_id, num_parts=num_parts)
        parts[part].append(normalize_chunk_id(chunk_id))
    for part in parts:
        parts[part] = sorted(set(parts[part]), key=int)
    return parts


def parts_document() -> str:
    """Identical ``PARTS.md`` body shipped in both MDC archives."""
    return (
        "SPDMX is published on Mozilla Data Collective as two archives.\n"
        "\n"
        "Part 1 contains even chunk ids (`chunk_0`, `chunk_2`, …).\n"
        "Part 2 contains odd chunk ids (`chunk_1`, `chunk_3`, …).\n"
        "\n"
        "Download both archives and extract them into the same directory:\n"
        "\n"
        "    mkdir -p SPDMX\n"
        "    tar -xzf spdmx-part1.tar.gz -C SPDMX\n"
        "    tar -xzf spdmx-part2.tar.gz -C SPDMX\n"
        "\n"
        "The CSV indexes describe the full corpus. `chunk_assignment.csv` "
        "records each song's chunk id. Later releases keep those songs in "
        "the same chunk, so even chunk ids stay in part 1 and odd chunk ids "
        "stay in part 2.\n"
        "\n"
        "Optionally run `./link_single_track_mixes.sh` after both extracts.\n"
    )


def assign_songs_to_chunks(
    song_sizes: Mapping[str, int],
    *,
    num_chunks: int = DEFAULT_NUM_CHUNKS,
    seed: int = CHUNK_ASSIGNMENT_SEED,
    frozen: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Pack songs into ``num_chunks`` roughly equal-sized bins (LPT).

    After a seeded shuffle (tie-break), songs are placed largest-first into the
    currently lightest chunk so loads stay balanced. Returns
    ``{song_id: chunk_id}`` with unpadded chunk ids (``0``, ``1``, …).

    The sort key is ``(-size, song_id)``, a total order, so the shuffle does
    not change the result when every song id is unique.

    If ``frozen`` is set, those songs keep their chunk ids even when sizes
    change. Only songs absent from ``frozen`` are packed, starting from the
    byte load of frozen songs that are present. Frozen songs that are missing
    from ``song_sizes`` are omitted from the result and are not reassigned.
    With no frozen map and fewer songs than ``num_chunks``, uses one chunk
    per song.
    """
    import heapq

    if num_chunks <= 0:
        raise ValueError(f"num_chunks must be > 0, got {num_chunks}")
    if not song_sizes:
        return {}

    for song_id, size in song_sizes.items():
        if int(size) < 0:
            raise ValueError(f"negative size for song {song_id}: {size}")

    frozen_norm: dict[str, str] = {}
    if frozen:
        for song_id, chunk_id in frozen.items():
            normalized = normalize_chunk_id(chunk_id)
            if int(normalized) >= int(num_chunks):
                raise ValueError(
                    f"frozen chunk {normalized} for song {song_id} is outside "
                    f"0..{int(num_chunks) - 1}"
                )
            frozen_norm[str(song_id)] = normalized

    assignment: dict[str, str] = {}
    if frozen_norm:
        loads_by_index = [0] * int(num_chunks)
        for song_id, size in song_sizes.items():
            key = str(song_id)
            if key not in frozen_norm:
                continue
            idx = int(frozen_norm[key])
            assignment[key] = format_chunk_id(idx)
            loads_by_index[idx] += int(size)
        items = [
            (str(song_id), int(song_sizes[song_id]))
            for song_id in song_sizes
            if str(song_id) not in frozen_norm
        ]
        n = int(num_chunks)
        loads: list[tuple[int, int]] = [
            (loads_by_index[i], i) for i in range(n)
        ]
    else:
        items = [(str(s), int(song_sizes[s])) for s in song_sizes]
        n = min(int(num_chunks), len(items))
        loads = [(0, i) for i in range(n)]

    if not items:
        return assignment

    rng = random.Random(seed)
    rng.shuffle(items)
    # Longest-processing-time: largest first; song_id breaks remaining ties.
    items.sort(key=lambda pair: (-pair[1], pair[0]))

    heapq.heapify(loads)
    for song_id, size in items:
        cur, idx = heapq.heappop(loads)
        assignment[song_id] = format_chunk_id(idx)
        heapq.heappush(loads, (cur + size, idx))

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
    out["mix"] = [
        packaged_mix_rel(chunk, song)
        for song, chunk in zip(song_ids, chunks, strict=True)
    ]
    # Stable column order for the packaged product.
    extra = [c for c in out.columns if c not in PACKAGED_TRACK_MAP_COLUMNS]
    return out[PACKAGED_TRACK_MAP_COLUMNS + extra]


def rewrite_songs_table_for_chunks(
    songs: pd.DataFrame,
    song_to_chunk: Mapping[str, str],
) -> pd.DataFrame:
    """Remap flat ``songs.csv`` paths into packaged ``chunk_N/...`` locations.

    Keeps ``song_length``, subset flags, and other song-level columns — no
    mix FLAC re-reads. Used by ``build_spdmx`` when ``SPDMX_dev/songs.csv``
    was already written by the mix pass.
    """
    if "song_id" not in songs.columns:
        raise ValueError("songs.csv missing required column: song_id")
    missing = sorted(set(songs["song_id"].astype(str)) - set(song_to_chunk))
    if missing:
        preview = ", ".join(missing[:5])
        more = "" if len(missing) <= 5 else f" (+{len(missing) - 5} more)"
        raise ValueError(f"No chunk assignment for song_id(s): {preview}{more}")

    out = songs.copy()
    song_ids = out["song_id"].astype(str)
    chunks = song_ids.map(lambda s: normalize_chunk_id(song_to_chunk[s]))
    out["chunk"] = chunks
    if "path" in out.columns:
        out["path"] = [
            packaged_audio_rel(chunk, song)
            for song, chunk in zip(song_ids, chunks, strict=True)
        ]
    if "mid" in out.columns:
        out["mid"] = [
            packaged_mid_rel(chunk, song)
            for song, chunk in zip(song_ids, chunks, strict=True)
        ]
    if "mix" in out.columns:
        out["mix"] = [
            packaged_mix_rel(chunk, song)
            for song, chunk in zip(song_ids, chunks, strict=True)
        ]
    # Derive subset:multitrack when remapping an older songs.csv that lacks it.
    if SPDMX_MULTITRACK_SUBSET_COLUMN not in out.columns and "n_tracks" in out.columns:
        out[SPDMX_MULTITRACK_SUBSET_COLUMN] = (
            pd.to_numeric(out["n_tracks"], errors="coerce").fillna(0).astype(int) >= 2
        )
    return out


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
