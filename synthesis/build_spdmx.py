"""Post-render packaging: flat ``SPDMX_dev/`` → chunked ``SPDMX/`` release tree.

Does **not** run synthesis and does **not** mutate the flat production tree.
After ``synthesis.final`` has written ``audio/``, ``mid/``, ``mix/``,
``stems.csv``, and ``songs.csv`` under ``SPDMX_dev/``, this script builds a
**separate** distributable directory (default ``{OUTPUT_DIR}/SPDMX/``):

1. Assigns songs to a fixed number of roughly equal-sized download chunks
   (default 64, LPT bin packing).
2. Hardlinks (or copies) stems + mix audio + dense MIDI into
   ``chunk_N/<song_id>/{k.flac,mix.flac,mix.mid}``.
3. Writes packaged ``stems.csv`` (with ``chunk``) + ``chunks.csv`` + LICENSE/README.
4. Packages ``songs.csv`` by remapping ``SPDMX_dev/songs.csv`` into chunk paths
   (preferred; keeps ``song_length``). Falls back to aggregating from packaged
   ``stems.csv`` if the dev songs table is missing.

The flat ``{OUTPUT_DIR}/SPDMX_dev/`` render stays intact for lab use.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from analysis.corrected_midi import TRACK_MAP_FILE_NAME
from shared.config import (
    OUTPUT_DIR,
    SPDMX_AUDIO_DIR_NAME,
    SPDMX_FILE_NAME,
    SPDMX_MID_DIR_NAME,
    SPDMX_MIX_DIR_NAME,
    SPDMX_RELEASE_MIX_AUDIO_NAME,
    SPDMX_RELEASE_MIX_MIDI_NAME,
    SPDMX_SONGS_FILE_NAME,
)
from synthesis.build_songs_table import write_songs_table
from synthesis.chunking import (
    CHUNK_ASSIGNMENT_SEED,
    DEFAULT_NUM_CHUNKS,
    CHUNKS_FILE_NAME,
    assign_songs_to_chunks,
    build_chunks_manifest,
    chunk_dir_name,
    list_chunk_dirs,
    rewrite_songs_table_for_chunks,
    rewrite_track_map_for_chunks,
    song_media_bytes,
)
from synthesis.paths import spdmx_dataset_dir, spdmx_dev_dir
from synthesis.spdmx_release import write_spdmx_release_docs


def parse_args(args=None, namespace=None):
    parser = argparse.ArgumentParser(
        description=(
            "Build a chunked sPDMX release tree in SPDMX/ from flat SPDMX_dev/ "
            "(production render is left untouched)."
        ),
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        default=OUTPUT_DIR,
        type=str,
        help="Pipeline output root (contains SPDMX_dev/).",
    )
    parser.add_argument(
        "--dataset-dir",
        default=None,
        type=str,
        help="Flat rendered tree (default: {output_dir}/SPDMX_dev).",
    )
    parser.add_argument(
        "--package-dir",
        default=None,
        type=str,
        help=(
            "New directory for chunk_N/ + packaged CSVs "
            "(default: {output_dir}/SPDMX)."
        ),
    )
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=DEFAULT_NUM_CHUNKS,
        help=(
            "Number of roughly equal-sized download chunks "
            f"(default: {DEFAULT_NUM_CHUNKS})."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=CHUNK_ASSIGNMENT_SEED,
        help=f"Shuffle seed for song→chunk assignment (default: {CHUNK_ASSIGNMENT_SEED}).",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of hardlinking into the package tree.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Assign chunks and print summary without writing media or CSVs.",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=None,
        help=(
            "Parallel workers for size scan, publish, cleanup, and songs.csv "
            f"(default: {_default_jobs()})."
        ),
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help=(
            "Allow --package-dir to equal --dataset-dir (mutates the flat tree; "
            "not recommended)."
        ),
    )
    return parser.parse_args(args=args, namespace=namespace)


def _parallel_map(fn, items, *, jobs: int, desc: str):
    """Thread-pool map with a tqdm bar (I/O-bound hardlink/stat work)."""
    items = list(items)
    n_jobs = max(1, int(jobs))
    label = desc if n_jobs <= 1 else f"{desc} (-j {n_jobs})"
    if n_jobs <= 1 or len(items) <= 1:
        return [fn(item) for item in tqdm(items, total=len(items), desc=label)]
    chunksize = max(1, min(64, len(items) // (n_jobs * 8) or 1))
    with ThreadPoolExecutor(max_workers=n_jobs) as pool:
        return list(
            tqdm(
                pool.map(fn, items, chunksize=chunksize),
                total=len(items),
                desc=label,
            )
        )


def _default_jobs() -> int:
    """Prefer more workers for NFS hardlink/stat throughput."""
    return max(8, min(32, (os.cpu_count() or 8) * 2))


def _measure_sizes_parallel(
    song_ids: list[str],
    *,
    audio_root: Path,
    mid_root: Path,
    mix_root: Path,
    jobs: int,
) -> dict[str, int]:
    def _one(song_id: str) -> tuple[str, int]:
        return song_id, song_media_bytes(
            song_id,
            audio_root=audio_root,
            mid_root=mid_root,
            mix_root=mix_root,
        )

    pairs = _parallel_map(_one, song_ids, jobs=jobs, desc="Measure song sizes")
    return {song_id: size for song_id, size in pairs}


def _load_flat_track_map(dataset_dir: Path) -> pd.DataFrame:
    csv_path = dataset_dir / TRACK_MAP_FILE_NAME
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Missing {csv_path}; run prepare_synthesis / synthesis.final first."
        )
    table = pd.read_csv(csv_path)
    if "song_id" not in table.columns:
        raise ValueError(f"{csv_path} missing song_id column")
    if "chunk" in table.columns:
        table = table.drop(columns=["chunk"])
    return table


def _file_inventory(root: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    if not root.is_dir():
        return out
    for path in root.rglob("*"):
        if path.is_file():
            out[path.relative_to(root).as_posix()] = path.stat().st_size
    return out


def _inventories_match(src: Path, dst: Path) -> bool:
    return _file_inventory(src) == _file_inventory(dst)


def _inventory_covers(src: Path, dst: Path) -> bool:
    """True if every file in *src* exists in *dst* with the same size.

    Dest may contain extras (e.g. ``mix.flac`` / ``mix.mid`` beside stems).
    """
    src_inv = _file_inventory(src)
    dst_inv = _file_inventory(dst)
    return all(dst_inv.get(rel) == size for rel, size in src_inv.items())


def _mirror_files(src: Path, dst: Path, *, copy: bool) -> None:
    """Hardlink (or copy) every file from *src* into *dst* without deleting *src*."""
    dst.mkdir(parents=True, exist_ok=True)
    for rel, _size in sorted(_file_inventory(src).items()):
        source_file = src / rel
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            try:
                if target.samefile(source_file):
                    continue
            except OSError:
                pass
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        if copy:
            shutil.copy2(source_file, target)
            continue
        try:
            os.link(source_file, target)
        except OSError:
            shutil.copy2(source_file, target)


def _publish_dir(src: Path, dst: Path, *, copy: bool) -> None:
    """Mirror *src* → *dst* and verify; never deletes *src*.

    Verification is cover-based so flattened song dirs may already contain
    ``mix.flac`` / ``mix.mid`` beside the stem FLACs.
    """
    if not src.exists():
        raise FileNotFoundError(f"missing source: {src}")
    if src.resolve() == dst.resolve():
        return
    if _inventory_covers(src, dst):
        return
    _mirror_files(src, dst, copy=copy)
    if not _inventory_covers(src, dst):
        raise RuntimeError(
            f"publish verify failed for {src} → {dst}: "
            f"source={_file_inventory(src)} dest={_file_inventory(dst)}"
        )


def _prune_empty_parents(path: Path, *, stop_at: Path) -> None:
    stop = stop_at.resolve()
    current = path.resolve()
    if not current.exists():
        current = current.parent
    while current != stop and stop in current.parents:
        if current.exists() and current.is_dir() and not any(current.iterdir()):
            current.rmdir()
            current = current.parent
            continue
        break


def _publish_file(src: Path, dst: Path, *, copy: bool) -> None:
    if not src.is_file():
        raise FileNotFoundError(f"missing source file: {src}")
    if dst.exists():
        try:
            if dst.samefile(src):
                return
        except OSError:
            pass
        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copy2(src, dst)
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)
    if dst.stat().st_size != src.stat().st_size:
        raise RuntimeError(f"publish verify failed for file {src} → {dst}")


def _place_song_media(
    song_id: str,
    chunk_id: str,
    *,
    source_audio: Path,
    source_mid: Path,
    source_mix: Path,
    package_dir: Path,
    copy: bool,
) -> None:
    """Publish stems + mix.flac + mix.mid into ``chunk_N/<song_id>/``."""
    song_dst = package_dir / chunk_dir_name(chunk_id) / song_id
    audio_src = source_audio / song_id
    mid_src = source_mid / f"{song_id}.mid"
    mix_src = source_mix / f"{song_id}.flac"
    if not audio_src.is_dir():
        raise FileNotFoundError(f"Missing flat audio for song_id={song_id}: {audio_src}")
    _publish_dir(audio_src, song_dst, copy=copy)
    if mid_src.is_file():
        _publish_file(
            mid_src,
            song_dst / SPDMX_RELEASE_MIX_MIDI_NAME,
            copy=copy,
        )
    if mix_src.is_file():
        _publish_file(
            mix_src,
            song_dst / SPDMX_RELEASE_MIX_AUDIO_NAME,
            copy=copy,
        )


def _is_song_media_dir(path: Path) -> bool:
    """True when *path* looks like a packaged song folder (stems and/or mix)."""
    if not path.is_dir():
        return False
    try:
        children = list(path.iterdir())
    except OSError:
        return False
    for child in children:
        if not child.is_file():
            continue
        if child.name in (SPDMX_RELEASE_MIX_AUDIO_NAME, SPDMX_RELEASE_MIX_MIDI_NAME):
            return True
        if child.suffix.lower() == ".flac" and child.stem.isdigit():
            return True
    return False


def _iter_packaged_song_dirs(chunk_dir: Path):
    """Yield song media dirs under ``chunk_N/<a>/<b>/<hash>/`` (depth-3 song_ids)."""
    try:
        top = list(chunk_dir.iterdir())
    except OSError:
        return
    for a in top:
        if not a.is_dir():
            continue
        if a.name in (SPDMX_AUDIO_DIR_NAME, SPDMX_MID_DIR_NAME):
            continue
        try:
            mid_level = list(a.iterdir())
        except OSError:
            continue
        for b in mid_level:
            if not b.is_dir():
                continue
            try:
                leaves = list(b.iterdir())
            except OSError:
                continue
            for c in leaves:
                if c.is_dir() and _is_song_media_dir(c):
                    yield c


def _cleanup_one_chunk_dir(
    chunk_dir: Path,
    keep: set[str] | None,
) -> dict[str, int | str]:
    """Clean one ``chunk_*`` tree.

    If *keep* is ``None``, the entire chunk directory is removed (obsolete
    chunk id). Otherwise drop legacy wrappers and song dirs not in *keep*.
    """
    name = chunk_dir.name
    if keep is None:
        print(f"  cleanup {name}: removing obsolete chunk directory …", flush=True)
        shutil.rmtree(chunk_dir)
        return {"chunk": name, "removed_songs": -1, "removed_legacy": 0, "kept": 0}

    removed_songs = 0
    removed_legacy = 0
    for legacy in (SPDMX_AUDIO_DIR_NAME, SPDMX_MID_DIR_NAME):
        legacy_path = chunk_dir / legacy
        if legacy_path.is_dir():
            shutil.rmtree(legacy_path)
            removed_legacy += 1

    print(f"  cleanup {name}: scanning for obsolete song dirs …", flush=True)
    to_remove: list[Path] = []
    for path in _iter_packaged_song_dirs(chunk_dir):
        try:
            rel = path.relative_to(chunk_dir).as_posix()
        except ValueError:
            continue
        if rel in keep:
            continue
        to_remove.append(path)

    if to_remove:
        print(
            f"  cleanup {name}: removing {len(to_remove)} obsolete song dir(s) …",
            flush=True,
        )
        # Parallel rmtree within the chunk (independent song trees).
        n_rm = min(8, max(1, len(to_remove)))
        if n_rm <= 1:
            for path in to_remove:
                shutil.rmtree(path)
                _prune_empty_parents(path.parent, stop_at=chunk_dir)
                removed_songs += 1
        else:
            with ThreadPoolExecutor(max_workers=n_rm) as pool:
                list(pool.map(shutil.rmtree, to_remove))
            for path in to_remove:
                _prune_empty_parents(path.parent, stop_at=chunk_dir)
                removed_songs += 1

    print(
        f"  cleanup {name}: done "
        f"(kept={len(keep)}, removed_songs={removed_songs}, "
        f"removed_legacy={removed_legacy})",
        flush=True,
    )
    return {
        "chunk": name,
        "removed_songs": removed_songs,
        "removed_legacy": removed_legacy,
        "kept": len(keep),
    }


def _cleanup_obsolete_chunk_dirs(
    package_dir: Path,
    assignment: dict[str, str],
    *,
    jobs: int = 8,
) -> None:
    """Drop stale chunk trees / song dirs left from a prior packing.

    Parallelized per ``chunk_*`` directory (I/O-bound NFS walks).
    """
    songs_by_chunk: dict[str, set[str]] = {}
    for song_id, chunk_id in assignment.items():
        songs_by_chunk.setdefault(chunk_dir_name(chunk_id), set()).add(song_id)

    chunk_dirs = list_chunk_dirs(package_dir)
    tasks: list[tuple[Path, set[str] | None]] = []
    for chunk_dir in chunk_dirs:
        keep = songs_by_chunk.get(chunk_dir.name)
        tasks.append((chunk_dir, keep))

    n_jobs = max(1, int(jobs))
    print(
        f"Cleaning obsolete paths under {package_dir} "
        f"({len(tasks)} chunk dir(s), -j {n_jobs}) …",
        flush=True,
    )

    def _one(task: tuple[Path, set[str] | None]) -> dict[str, int | str]:
        chunk_dir, keep = task
        return _cleanup_one_chunk_dir(chunk_dir, keep)

    results = _parallel_map(_one, tasks, jobs=n_jobs, desc="Cleanup chunk dirs")
    n_removed = sum(int(r["removed_songs"]) for r in results if int(r["removed_songs"]) > 0)
    n_dropped_chunks = sum(1 for r in results if int(r["removed_songs"]) < 0)
    print(
        f"Cleanup finished: removed {n_removed} obsolete song dir(s); "
        f"dropped {n_dropped_chunks} empty/obsolete chunk dir(s).",
        flush=True,
    )


def chunk_dataset(
    *,
    dataset_dir: str | Path,
    package_dir: str | Path,
    num_chunks: int = DEFAULT_NUM_CHUNKS,
    seed: int = CHUNK_ASSIGNMENT_SEED,
    copy: bool = False,
    dry_run: bool = False,
    allow_in_place: bool = False,
    jobs: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Build a chunked release tree under *package_dir* from flat *dataset_dir*.

    Flat ``audio/`` / ``mid/`` / ``mix/`` under *dataset_dir* are never deleted.
    """
    source = Path(dataset_dir)
    dest = Path(package_dir)
    if source.resolve() == dest.resolve() and not allow_in_place:
        raise ValueError(
            f"Refusing to package in-place into {source}. "
            f"Pass a distinct --package-dir (default SPDMX/) "
            f"or --in-place if you really mean it."
        )

    dest.mkdir(parents=True, exist_ok=True)

    table = _load_flat_track_map(source)
    song_ids = sorted(table["song_id"].astype(str).unique())
    audio_root = source / SPDMX_AUDIO_DIR_NAME
    mid_root = source / SPDMX_MID_DIR_NAME
    mix_root = source / SPDMX_MIX_DIR_NAME
    if not audio_root.is_dir():
        raise FileNotFoundError(f"Missing flat audio tree: {audio_root}")
    if not mid_root.is_dir():
        raise FileNotFoundError(f"Missing flat mid tree: {mid_root}")

    song_sizes = _measure_sizes_parallel(
        song_ids,
        audio_root=audio_root,
        mid_root=mid_root,
        mix_root=mix_root,
        jobs=jobs,
    )

    def _audio_missing(song_id: str) -> str | None:
        return song_id if not (audio_root / song_id).is_dir() else None

    print(f"Checking flat audio presence ({len(song_ids)} songs) …", flush=True)
    missing_flags = _parallel_map(
        _audio_missing, song_ids, jobs=jobs, desc="Check audio dirs",
    )
    missing_audio = [s for s in missing_flags if s is not None]
    if missing_audio:
        preview = ", ".join(missing_audio[:5])
        more = "" if len(missing_audio) <= 5 else f" (+{len(missing_audio) - 5} more)"
        raise FileNotFoundError(f"Missing audio for song_id(s): {preview}{more}")

    assignment = assign_songs_to_chunks(
        song_sizes, num_chunks=num_chunks, seed=seed,
    )
    packaged = rewrite_track_map_for_chunks(table, assignment)
    chunks = build_chunks_manifest(packaged, assignment, song_sizes)

    if dry_run:
        return packaged, chunks, assignment

    items = list(assignment.items())

    def _place(pair: tuple[str, str]) -> None:
        song_id, chunk_id = pair
        _place_song_media(
            song_id,
            chunk_id,
            source_audio=audio_root,
            source_mid=mid_root,
            source_mix=mix_root,
            package_dir=dest,
            copy=copy,
        )

    _parallel_map(_place, items, jobs=jobs, desc="Publish into chunks")

    print("Publish finished; starting obsolete-path cleanup …", flush=True)
    _cleanup_obsolete_chunk_dirs(dest, assignment, jobs=jobs)

    print(f"Writing packaged CSVs under {dest} …", flush=True)
    packaged.to_csv(dest / f"{SPDMX_FILE_NAME}.csv", index=False)
    chunks.to_csv(dest / CHUNKS_FILE_NAME, index=False)
    write_spdmx_release_docs(dest)

    source_songs = source / SPDMX_SONGS_FILE_NAME
    dest_songs = dest / SPDMX_SONGS_FILE_NAME
    if source_songs.is_file():
        print(
            f"Packaging songs.csv from {source_songs} (remap paths; "
            "no mix FLAC re-reads) …",
            flush=True,
        )
        songs = rewrite_songs_table_for_chunks(
            pd.read_csv(source_songs), assignment,
        )
        songs.to_csv(dest_songs, index=False)
        summary = {
            "path": str(dest_songs),
            "n_songs": int(len(songs)),
            "n_with_song_length": (
                int(songs["song_length"].notna().sum())
                if "song_length" in songs.columns
                else 0
            ),
            "source": str(source_songs),
        }
        with open(dest_songs.with_suffix(".summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
    else:
        # Media just published; trust packaged stems.csv paths (no second disk scan).
        print(
            f"No {source_songs.name} under {source}; building songs.csv "
            "from packaged stems.csv …",
            flush=True,
        )
        write_songs_table(dest, check_files=False, jobs=jobs)
    print("Chunk packaging complete.", flush=True)
    return packaged, chunks, assignment

def main(argv=None) -> int:
    args = parse_args(argv)
    dataset_dir = Path(
        args.dataset_dir
        if args.dataset_dir is not None
        else spdmx_dev_dir(args.output_dir)
    )
    package_dir = Path(
        args.package_dir
        if args.package_dir is not None
        else spdmx_dataset_dir(args.output_dir)
    )

    try:
        packaged, chunks, assignment = chunk_dataset(
            dataset_dir=dataset_dir,
            package_dir=package_dir,
            num_chunks=args.num_chunks,
            seed=args.seed,
            copy=args.copy,
            dry_run=args.dry_run,
            allow_in_place=args.in_place,
            jobs=args.jobs if args.jobs is not None else _default_jobs(),
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    n_songs = len(assignment)
    n_chunks = 0 if chunks.empty else len(chunks)
    total_bytes = int(chunks["bytes"].sum()) if n_chunks else 0
    mode = "dry-run" if args.dry_run else "wrote"
    print(
        f"{mode}: {n_songs} songs → {n_chunks} chunks "
        f"({total_bytes / (1024**3):.2f} GiB media) "
        f"from {dataset_dir} → {package_dir}"
    )
    if not args.dry_run:
        print(f"  {package_dir / f'{SPDMX_FILE_NAME}.csv'}")
        print(f"  {package_dir / SPDMX_SONGS_FILE_NAME}")
        print(f"  {package_dir / CHUNKS_FILE_NAME}")
        print(f"  flat production tree left untouched: {dataset_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
