"""Post-render packaging: flat ``SPDMX_dev/`` → chunked ``SPDMX/`` release tree.

Does **not** run synthesis and does **not** mutate the flat production tree.
After ``synthesis.final`` has written ``audio/``, ``mid/``, and ``SPDMX.csv``
under ``SPDMX_dev/``, this script builds a **separate** distributable directory
(default ``{OUTPUT_DIR}/SPDMX/``):

1. Assigns songs to ~25 GiB download chunks.
2. Hardlinks (or copies) ``audio/`` and ``mid/`` into ``chunk_N/{audio,mid}/``.
3. Writes packaged ``SPDMX.csv`` (with ``chunk``) + ``chunks.csv`` + LICENSE/README.

The flat ``{OUTPUT_DIR}/SPDMX_dev/`` render stays intact for lab use.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from analysis.corrected_midi import TRACK_MAP_FILE_NAME
from shared.config import (
    OUTPUT_DIR,
    SPDMX_AUDIO_DIR_NAME,
    SPDMX_FILE_NAME,
    SPDMX_MID_DIR_NAME,
)
from synthesis.chunking import (
    CHUNK_ASSIGNMENT_SEED,
    CHUNK_BYTES_TARGET,
    CHUNKS_FILE_NAME,
    assign_songs_to_chunks,
    build_chunks_manifest,
    chunk_dir_name,
    list_chunk_dirs,
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
        "--target-bytes",
        type=int,
        default=CHUNK_BYTES_TARGET,
        help=f"Soft size budget per chunk in bytes (default: {CHUNK_BYTES_TARGET}).",
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
        default=8,
        help="Parallel workers for size scan + hardlink/copy (default: 8).",
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
    results = [None] * len(items)
    with ThreadPoolExecutor(max_workers=n_jobs) as pool:
        futures = {pool.submit(fn, item): i for i, item in enumerate(items)}
        with tqdm(total=len(items), desc=label) as pbar:
            for fut in as_completed(futures):
                i = futures[fut]
                results[i] = fut.result()
                pbar.update(1)
    return results


def _measure_sizes_parallel(
    song_ids: list[str],
    *,
    audio_root: Path,
    mid_root: Path,
    jobs: int,
) -> dict[str, int]:
    def _one(song_id: str) -> tuple[str, int]:
        return song_id, song_media_bytes(
            song_id, audio_root=audio_root, mid_root=mid_root,
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
    """Mirror *src* → *dst* and verify; never deletes *src*."""
    if not src.exists():
        raise FileNotFoundError(f"missing source: {src}")
    if src.resolve() == dst.resolve():
        return
    if _inventories_match(src, dst):
        return
    _mirror_files(src, dst, copy=copy)
    if not _inventories_match(src, dst):
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
    package_dir: Path,
    copy: bool,
) -> None:
    chunk_root = package_dir / chunk_dir_name(chunk_id)
    audio_src = source_audio / song_id
    mid_src = source_mid / f"{song_id}.mid"
    if not audio_src.is_dir():
        raise FileNotFoundError(f"Missing flat audio for song_id={song_id}: {audio_src}")
    _publish_dir(
        audio_src,
        chunk_root / SPDMX_AUDIO_DIR_NAME / song_id,
        copy=copy,
    )
    if mid_src.is_file():
        _publish_file(
            mid_src,
            chunk_root / SPDMX_MID_DIR_NAME / f"{song_id}.mid",
            copy=copy,
        )


def _cleanup_obsolete_chunk_dirs(
    package_dir: Path,
    assignment: dict[str, str],
) -> None:
    songs_by_chunk: dict[str, set[str]] = {}
    for song_id, chunk_id in assignment.items():
        songs_by_chunk.setdefault(chunk_dir_name(chunk_id), set()).add(song_id)

    for chunk_dir in list_chunk_dirs(package_dir):
        if chunk_dir.name not in songs_by_chunk:
            shutil.rmtree(chunk_dir)
            continue
        audio_root = chunk_dir / SPDMX_AUDIO_DIR_NAME
        if not audio_root.is_dir():
            continue
        keep = songs_by_chunk[chunk_dir.name]
        for path in sorted(audio_root.rglob("*"), reverse=True):
            if not path.is_dir():
                continue
            try:
                rel = path.relative_to(audio_root).as_posix()
            except ValueError:
                continue
            if rel.count("/") != 2:
                continue
            if rel not in keep:
                shutil.rmtree(path)
                _prune_empty_parents(path.parent, stop_at=audio_root)


def chunk_dataset(
    *,
    dataset_dir: str | Path,
    package_dir: str | Path,
    target_bytes: int = CHUNK_BYTES_TARGET,
    seed: int = CHUNK_ASSIGNMENT_SEED,
    copy: bool = False,
    dry_run: bool = False,
    allow_in_place: bool = False,
    jobs: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Build a chunked release tree under *package_dir* from flat *dataset_dir*.

    Flat ``audio/`` / ``mid/`` under *dataset_dir* are never deleted.
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
    if not audio_root.is_dir():
        raise FileNotFoundError(f"Missing flat audio tree: {audio_root}")
    if not mid_root.is_dir():
        raise FileNotFoundError(f"Missing flat mid tree: {mid_root}")

    song_sizes = _measure_sizes_parallel(
        song_ids, audio_root=audio_root, mid_root=mid_root, jobs=jobs,
    )
    missing_audio = [
        s for s in song_ids if not (audio_root / s).is_dir()
    ]
    if missing_audio:
        preview = ", ".join(missing_audio[:5])
        more = "" if len(missing_audio) <= 5 else f" (+{len(missing_audio) - 5} more)"
        raise FileNotFoundError(f"Missing audio for song_id(s): {preview}{more}")

    assignment = assign_songs_to_chunks(
        song_sizes, target_bytes=target_bytes, seed=seed,
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
            package_dir=dest,
            copy=copy,
        )

    _parallel_map(_place, items, jobs=jobs, desc="Publish into chunks")

    _cleanup_obsolete_chunk_dirs(dest, assignment)

    packaged.to_csv(dest / f"{SPDMX_FILE_NAME}.csv", index=False)
    chunks.to_csv(dest / CHUNKS_FILE_NAME, index=False)
    write_spdmx_release_docs(dest)
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
            target_bytes=args.target_bytes,
            seed=args.seed,
            copy=args.copy,
            dry_run=args.dry_run,
            allow_in_place=args.in_place,
            jobs=args.jobs,
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
        print(f"  {package_dir / CHUNKS_FILE_NAME}")
        print(f"  flat production tree left untouched: {dataset_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
