"""Stage Zenodo-ready sPDMX artifacts from a chunked dataset tree.

Input: packaged layout from ``python -m synthesis.build_spdmx``
(``SPDMX.csv`` with ``chunk``, ``chunks.csv``, ``chunk_NNN/`` dirs).

Output staging directory (separate downloadable files):

* ``LICENSE``, ``README.md``, ``SPDMX.csv``, ``chunks.csv``
* ``chunk_NNN.zip`` for each chunk (~25 GiB media)
* ``SHA256SUMS``

Does not upload to Zenodo.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import zipfile
from pathlib import Path

import pandas as pd

from shared.config import OUTPUT_DIR, SPDMX_FILE_NAME
from synthesis.chunking import (
    CHUNKS_CSV_COLUMNS,
    CHUNKS_FILE_NAME,
    SHA256SUMS_FILE_NAME,
    chunk_dir_name,
    list_chunk_dirs,
    normalize_chunk_id,
    parse_chunk_id,
)
from synthesis.paths import spdmx_dataset_dir
from synthesis.spdmx_release import RELEASE_DOC_NAMES


def parse_args(args=None, namespace=None):
    parser = argparse.ArgumentParser(
        description=(
            "Stage Zenodo upload files from a chunked sPDMX tree "
            "(loose metadata + one zip per chunk)."
        ),
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        default=OUTPUT_DIR,
        type=str,
        help="Pipeline output root (used when --dataset-dir is omitted).",
    )
    parser.add_argument(
        "--dataset-dir",
        default=None,
        type=str,
        help="Chunked SPDMX/ release tree (default: {output_dir}/SPDMX).",
    )
    parser.add_argument(
        "--stage-dir",
        required=True,
        type=str,
        help="Directory to write Zenodo-ready files into.",
    )
    parser.add_argument(
        "--chunks",
        default=None,
        type=str,
        help="Comma-separated chunk ids to stage (default: all), e.g. 0,12.",
    )
    parser.add_argument(
        "--compression",
        choices=("store", "deflate"),
        default="store",
        help=(
            "Zip compression. Default store (FLAC already compressed); "
            "deflate is slower and rarely smaller."
        ),
    )
    return parser.parse_args(args=args, namespace=namespace)


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _parse_chunk_filter(raw: str | None) -> set[str] | None:
    if raw is None or not str(raw).strip():
        return None
    ids = set()
    for part in str(raw).split(","):
        token = part.strip()
        if not token:
            continue
        ids.add(normalize_chunk_id(token))
    return ids


def _zip_directory(source_dir: Path, zip_path: Path, *, compression: int) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()
    # Archive members are relative to the dataset root so unzipping beside
    # SPDMX.csv reconstructs ./chunk_NNN/audio/... paths.
    root = source_dir.parent
    with zipfile.ZipFile(zip_path, "w", compression=compression) as zf:
        for path in sorted(source_dir.rglob("*")):
            if path.is_file():
                zf.write(path, arcname=str(path.relative_to(root)))


def _write_sha256sums(stage_dir: Path, files: list[Path]) -> Path:
    lines = []
    for path in files:
        digest = _sha256_file(path)
        lines.append(f"{digest}  {path.name}")
    out = stage_dir / SHA256SUMS_FILE_NAME
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def stage_zenodo_files(
    *,
    dataset_dir: str | Path,
    stage_dir: str | Path,
    chunk_ids: set[str] | None = None,
    compression: str = "store",
) -> pd.DataFrame:
    """Stage metadata + chunk zips. Returns the updated ``chunks.csv`` table."""
    source = Path(dataset_dir)
    dest = Path(stage_dir)
    dest.mkdir(parents=True, exist_ok=True)

    spdmx_csv = source / f"{SPDMX_FILE_NAME}.csv"
    chunks_csv = source / CHUNKS_FILE_NAME
    if not spdmx_csv.is_file():
        raise FileNotFoundError(f"Missing {spdmx_csv}")
    if not chunks_csv.is_file():
        raise FileNotFoundError(
            f"Missing {chunks_csv}; run python -m synthesis.build_spdmx first."
        )

    for name in RELEASE_DOC_NAMES:
        src = source / name
        if not src.is_file():
            raise FileNotFoundError(f"Missing release doc: {src}")
        shutil.copy2(src, dest / name)

    shutil.copy2(spdmx_csv, dest / spdmx_csv.name)

    chunks = pd.read_csv(chunks_csv)
    for col in CHUNKS_CSV_COLUMNS:
        if col not in chunks.columns:
            chunks[col] = "" if col in ("archive", "sha256") else 0
    chunks = chunks[CHUNKS_CSV_COLUMNS].copy()
    chunks["chunk"] = chunks["chunk"].map(normalize_chunk_id)

    zip_compression = (
        zipfile.ZIP_STORED if compression == "store" else zipfile.ZIP_DEFLATED
    )
    available = {parse_chunk_id(p.name): p for p in list_chunk_dirs(source)}
    selected = (
        sorted(chunk_ids, key=int)
        if chunk_ids is not None
        else sorted(available, key=int)
    )
    if chunk_ids is not None:
        missing = sorted(set(chunk_ids) - set(available), key=int)
        if missing:
            raise FileNotFoundError(
                f"Requested chunk dir(s) not found under {source}: {missing}"
            )

    staged_files: list[Path] = [
        dest / name for name in RELEASE_DOC_NAMES
    ] + [dest / spdmx_csv.name]

    archive_by_chunk: dict[str, tuple[str, str, int]] = {}
    for chunk_id in selected:
        chunk_path = available[chunk_id]
        archive_name = f"{chunk_dir_name(chunk_id)}.zip"
        zip_path = dest / archive_name
        _zip_directory(chunk_path, zip_path, compression=zip_compression)
        digest = _sha256_file(zip_path)
        archive_by_chunk[chunk_id] = (archive_name, digest, zip_path.stat().st_size)
        staged_files.append(zip_path)

    # Update manifest rows for staged chunks; leave others untouched.
    archives = []
    digests = []
    for _, row in chunks.iterrows():
        chunk_id = normalize_chunk_id(row["chunk"])
        if chunk_id in archive_by_chunk:
            name, digest, _size = archive_by_chunk[chunk_id]
            archives.append(name)
            digests.append(digest)
        else:
            archives.append(row.get("archive", "") or "")
            digests.append(row.get("sha256", "") or "")
    chunks["archive"] = archives
    chunks["sha256"] = digests

    if chunk_ids is not None:
        # Staging a subset: still ship full SPDMX.csv, but chunks.csv lists
        # only the zips present in this stage directory.
        chunks = chunks[chunks["chunk"].isin(selected)].reset_index(drop=True)

    staged_chunks = dest / CHUNKS_FILE_NAME
    chunks.to_csv(staged_chunks, index=False)
    staged_files.append(staged_chunks)

    _write_sha256sums(dest, staged_files)
    return chunks


def main(argv=None) -> int:
    args = parse_args(argv)
    dataset_dir = Path(
        args.dataset_dir
        if args.dataset_dir is not None
        else spdmx_dataset_dir(args.output_dir)
    )
    stage_dir = Path(args.stage_dir)
    try:
        chunks = stage_zenodo_files(
            dataset_dir=dataset_dir,
            stage_dir=stage_dir,
            chunk_ids=_parse_chunk_filter(args.chunks),
            compression=args.compression,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    n_zips = int((chunks["archive"].astype(str) != "").sum())
    print(
        f"staged {n_zips} chunk zip(s) + metadata into {stage_dir} "
        f"(from {dataset_dir})"
    )
    print(f"  {stage_dir / SHA256SUMS_FILE_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
