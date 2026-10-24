"""Stage Mozilla Data Collective upload archives from a chunked SPDMX tree.

Input: packaged layout from ``python -m synthesis.build_spdmx``
(``stems.csv`` / ``songs.csv`` / ``chunks.csv``, ``chunk_N/`` dirs).

Output staging directory (default ``{OUTPUT}/SPDMX_mdc/``):

* ``spdmx-part1.tar.gz`` / ``spdmx-part2.tar.gz`` — even / odd chunk ids
  (default). Both archives carry the full CSV index. ``--parts 1`` writes
  the legacy single ``spdmx.tar.gz`` instead.
* ``spdmx-sample.tar.gz`` — stratified song sample for the MDC preview slot
* ``SHA256SUMS``

Does not upload to Mozilla Data Collective. Multipart upload is handled by
the ``datacollective`` SDK after these archives exist.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from shared.config import (
    OUTPUT_DIR,
    SPDMX_FILE_NAME,
    SPDMX_MULTITRACK_SUBSET_COLUMN,
    SPDMX_SONGS_FILE_NAME,
)
from synthesis.chunking import (
    CHUNK_ASSIGNMENT_FILE_NAME,
    CHUNKS_FILE_NAME,
    DEFAULT_MAX_PART_BYTES,
    DEFAULT_NUM_MDC_PARTS,
    PARTS_FILE_NAME,
    SHA256SUMS_FILE_NAME,
    bundled_chunk_assignment_path,
    chunk_dir_name,
    normalize_chunk_id,
    partition_chunk_ids,
    parts_document,
)
from synthesis.paths import spdmx_dataset_dir
from synthesis.spdmx_release import RELEASE_DOC_NAMES

_TAR_PIPE_CHUNK = 8 * 1024 * 1024

FULL_ARCHIVE_NAME = "spdmx.tar.gz"
PART1_ARCHIVE_NAME = "spdmx-part1.tar.gz"
PART2_ARCHIVE_NAME = "spdmx-part2.tar.gz"
SAMPLE_ARCHIVE_NAME = "spdmx-sample.tar.gz"
SAMPLE_DIR_NAME = "spdmx-sample"
SAMPLE_NOTE_NAME = "SAMPLE.md"
DEFAULT_STAGE_DIR_NAME = "SPDMX_mdc"
DEFAULT_SAMPLE_N_MULTITRACK = 16
DEFAULT_SAMPLE_N_SINGLE = 16
DEFAULT_SAMPLE_SEED = 42


def parse_args(args=None, namespace=None):
    parser = argparse.ArgumentParser(
        description=(
            "Stage Mozilla Data Collective .tar.gz archives from a chunked "
            "sPDMX tree (two even/odd parts by default, plus an optional sample)."
        ),
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        default=OUTPUT_DIR,
        type=str,
        help="Pipeline output root (used when --dataset-dir / --stage-dir omitted).",
    )
    parser.add_argument(
        "--dataset-dir",
        default=None,
        type=str,
        help="Chunked SPDMX/ release tree (default: {output_dir}/SPDMX).",
    )
    parser.add_argument(
        "--stage-dir",
        default=None,
        type=str,
        help=(
            "Directory for MDC archives (default: {output_dir}/"
            f"{DEFAULT_STAGE_DIR_NAME})."
        ),
    )
    parser.add_argument(
        "--skip-full",
        action="store_true",
        help=(
            "Do not build the corpus archives "
            f"({PART1_ARCHIVE_NAME} and {PART2_ARCHIVE_NAME}, "
            f"or {FULL_ARCHIVE_NAME} when --parts 1)."
        ),
    )
    parser.add_argument(
        "--skip-sample",
        action="store_true",
        help=f"Do not build {SAMPLE_ARCHIVE_NAME}.",
    )
    parser.add_argument(
        "--sample-n-multitrack",
        type=int,
        default=DEFAULT_SAMPLE_N_MULTITRACK,
        help="Multitrack songs in the sample (default: %(default)s).",
    )
    parser.add_argument(
        "--sample-n-single",
        type=int,
        default=DEFAULT_SAMPLE_N_SINGLE,
        help="Single-track songs in the sample (default: %(default)s).",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=DEFAULT_SAMPLE_SEED,
        help="RNG seed for sample song selection (default: %(default)s).",
    )
    parser.add_argument(
        "--keep-sample-dir",
        action="store_true",
        help=f"Keep unpacked {SAMPLE_DIR_NAME}/ under the stage dir after tarring.",
    )
    parser.add_argument(
        "--parts",
        type=int,
        default=DEFAULT_NUM_MDC_PARTS,
        help=(
            "Corpus archives to write. 2 (default) splits chunk directories "
            f"into {PART1_ARCHIVE_NAME} (even chunk ids) and "
            f"{PART2_ARCHIVE_NAME} (odd chunk ids). 1 writes "
            f"{FULL_ARCHIVE_NAME} only."
        ),
    )
    parser.add_argument(
        "--max-part-bytes",
        type=int,
        default=DEFAULT_MAX_PART_BYTES,
        help=(
            "Refuse to archive a part whose raw media plus metadata exceeds "
            "this many bytes (default: 950 GB). Checked for --parts 2 only."
        ),
    )
    parser.add_argument(
        "--pigz-level",
        type=int,
        default=1,
        help=(
            "pigz / gzip compression level (default: 1). FLAC is already "
            "compressed; higher levels mainly slow the job."
        ),
    )
    return parser.parse_args(args=args, namespace=namespace)


def _sha256_file(
    path: Path,
    *,
    chunk_size: int = 1024 * 1024,
    desc: str | None = None,
) -> str:
    digest = hashlib.sha256()
    total = path.stat().st_size if path.is_file() else None
    label = desc or f"sha256 {path.name}"
    with path.open("rb") as handle:
        with tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=label,
            leave=False,
        ) as pbar:
            while True:
                block = handle.read(chunk_size)
                if not block:
                    break
                digest.update(block)
                pbar.update(len(block))
    return digest.hexdigest()


def _write_sha256sums(stage_dir: Path, files: list[Path]) -> Path:
    lines = []
    for path in files:
        digest = _sha256_file(path)
        lines.append(f"{digest}  {path.name}")
    out = stage_dir / SHA256SUMS_FILE_NAME
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def _release_relpath(path_value: str) -> Path:
    """Map CSV paths like ``./chunk_0/<song_id>`` to a path under the dataset root."""
    raw = Path(str(path_value))
    parts = raw.parts
    if parts and parts[0] == ".":
        return Path(*parts[1:]) if len(parts) > 1 else Path()
    return raw


def _compressor_cmd(level: int) -> list[str]:
    level = max(1, min(int(level), 9))
    if shutil.which("pigz"):
        return ["pigz", f"-{level}"]
    return ["gzip", f"-{level}"]


def _dir_byte_size(root: Path) -> int:
    total = 0
    for path in root.rglob("*"):
        if path.is_file() and not path.is_symlink():
            try:
                total += path.stat().st_size
            except OSError:
                continue
    return total


def estimate_tar_payload_bytes(source_dir: Path) -> int | None:
    """Best-effort uncompressed payload size for progress (excludes tar headers).

    Prefer ``chunks.csv`` ``bytes`` (fast on the full release) plus metadata
    file sizes. Fall back to a directory walk for small trees (samples).
    """
    source_dir = Path(source_dir)
    chunks_csv = source_dir / CHUNKS_FILE_NAME
    if chunks_csv.is_file():
        try:
            header = pd.read_csv(chunks_csv, nrows=0)
        except (ValueError, pd.errors.EmptyDataError):
            header = pd.DataFrame()
        if "bytes" in header.columns:
            chunks = pd.read_csv(chunks_csv, usecols=["bytes"])
            total = int(pd.to_numeric(chunks["bytes"], errors="coerce").fillna(0).sum())
            for name in (
                f"{SPDMX_FILE_NAME}.csv",
                SPDMX_SONGS_FILE_NAME,
                CHUNKS_FILE_NAME,
                SAMPLE_NOTE_NAME,
                *RELEASE_DOC_NAMES,
            ):
                path = source_dir / name
                if path.is_file():
                    total += path.stat().st_size
            return max(total, 1)
    # Small trees (sample): walking is cheap.
    walked = _dir_byte_size(source_dir)
    return walked if walked > 0 else None


def _tar_create_command(
    source_dir: Path,
    members: list[str] | None,
    extra_paths: list[Path] | None,
) -> list[str]:
    """``tar -cf -`` members. ``None`` members means the whole directory."""
    if members is None and not extra_paths:
        return ["tar", "-C", str(source_dir), "-cf", "-", "."]
    command = ["tar", "-cf", "-"]
    if members:
        for name in members:
            if name.startswith("/") or ".." in Path(name).parts:
                raise ValueError(f"unsafe tar member: {name}")
        command.extend(["-C", str(source_dir), *members])
    for path in extra_paths or []:
        command.extend(["-C", str(path.parent), path.name])
    return command


def stream_directory_to_tar_gz(
    source_dir: Path,
    archive_path: Path,
    *,
    pigz_level: int = 1,
    desc: str | None = None,
    total_bytes: int | None = None,
    members: list[str] | None = None,
    extra_paths: list[Path] | None = None,
) -> None:
    """Write ``source_dir`` contents to ``archive_path`` (archive root = contents).

    ``members`` lists top-level names under ``source_dir``. ``extra_paths`` are
    files from other directories, stored under their basename.
    """
    source_dir = source_dir.resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Not a directory: {source_dir}")
    archive_path = archive_path.resolve()
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_path.exists():
        archive_path.unlink()

    if total_bytes is None and members is None:
        total_bytes = estimate_tar_payload_bytes(source_dir)

    compressor = _compressor_cmd(pigz_level)
    label = desc or f"tar→{archive_path.name}"
    tar = subprocess.Popen(
        _tar_create_command(source_dir, members, extra_paths),
        stdout=subprocess.PIPE,
    )
    assert tar.stdout is not None
    pigz = subprocess.Popen(
        compressor,
        stdin=subprocess.PIPE,
        stdout=archive_path.open("wb"),
    )
    assert pigz.stdin is not None
    try:
        with tqdm(
            total=total_bytes,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=label,
            smoothing=0.05,
            miniters=1,
        ) as pbar:
            while True:
                block = tar.stdout.read(_TAR_PIPE_CHUNK)
                if not block:
                    break
                pigz.stdin.write(block)
                pbar.update(len(block))
                # Tar headers make the stream slightly larger than file bytes;
                # keep the bar from sticking at 100% early.
                if total_bytes is not None and pbar.n > total_bytes:
                    pbar.total = pbar.n
        pigz.stdin.close()
        tar_rc = tar.wait()
        pigz_rc = pigz.wait()
    except Exception:
        tar.kill()
        pigz.kill()
        archive_path.unlink(missing_ok=True)
        raise
    finally:
        if tar.stdout is not None:
            tar.stdout.close()
        if pigz.stdin is not None and not pigz.stdin.closed:
            pigz.stdin.close()
        if pigz.stdout is not None:
            pigz.stdout.close()

    if tar_rc != 0:
        archive_path.unlink(missing_ok=True)
        raise RuntimeError(f"tar failed with exit code {tar_rc}")
    if pigz_rc != 0:
        archive_path.unlink(missing_ok=True)
        raise RuntimeError(f"{' '.join(compressor)} failed with exit code {pigz_rc}")

def _require_release_tree(dataset_dir: Path) -> None:
    stems = dataset_dir / f"{SPDMX_FILE_NAME}.csv"
    songs = dataset_dir / SPDMX_SONGS_FILE_NAME
    chunks = dataset_dir / CHUNKS_FILE_NAME
    missing = [p for p in (stems, songs, chunks) if not p.is_file()]
    for name in RELEASE_DOC_NAMES:
        path = dataset_dir / name
        if not path.is_file():
            missing.append(path)
    if missing:
        raise FileNotFoundError(
            "Missing release file(s): "
            + ", ".join(str(p) for p in missing)
            + "; run python -m synthesis.build_spdmx first."
        )


def build_sample_tree(
    *,
    dataset_dir: Path,
    sample_dir: Path,
    n_multitrack: int,
    n_single: int,
    seed: int,
) -> pd.DataFrame:
    """Write a stratified sample tree; return the sampled ``songs`` frame."""
    if n_multitrack < 0 or n_single < 0:
        raise ValueError("sample sizes must be >= 0")
    if n_multitrack + n_single == 0:
        raise ValueError("need at least one sample song")

    songs = pd.read_csv(dataset_dir / SPDMX_SONGS_FILE_NAME)
    stems = pd.read_csv(dataset_dir / f"{SPDMX_FILE_NAME}.csv")
    chunks = pd.read_csv(dataset_dir / CHUNKS_FILE_NAME)

    if SPDMX_MULTITRACK_SUBSET_COLUMN not in songs.columns:
        raise ValueError(
            f"{SPDMX_SONGS_FILE_NAME} missing {SPDMX_MULTITRACK_SUBSET_COLUMN}"
        )
    if "path" not in songs.columns or "song_id" not in songs.columns:
        raise ValueError(f"{SPDMX_SONGS_FILE_NAME} missing song_id/path")

    mt_mask = songs[SPDMX_MULTITRACK_SUBSET_COLUMN].astype(bool)
    multitrack = songs[mt_mask]
    single = songs[~mt_mask]
    if len(multitrack) < n_multitrack:
        raise ValueError(
            f"requested {n_multitrack} multitrack songs but only {len(multitrack)} exist"
        )
    if len(single) < n_single:
        raise ValueError(
            f"requested {n_single} single-track songs but only {len(single)} exist"
        )

    pick = pd.concat(
        [
            multitrack.sample(n=n_multitrack, random_state=seed),
            single.sample(n=n_single, random_state=seed),
        ],
        ignore_index=True,
    )
    ids = set(pick["song_id"].astype(str))

    if sample_dir.exists():
        shutil.rmtree(sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)

    pick.to_csv(sample_dir / SPDMX_SONGS_FILE_NAME, index=False)
    stems[stems["song_id"].astype(str).isin(ids)].to_csv(
        sample_dir / f"{SPDMX_FILE_NAME}.csv", index=False
    )
    chunks_used = sorted(pick["chunk"].map(str).unique(), key=lambda x: int(x))
    chunk_col = chunks["chunk"].map(str)
    chunks[chunk_col.isin(chunks_used)].to_csv(
        sample_dir / CHUNKS_FILE_NAME, index=False
    )

    for name in RELEASE_DOC_NAMES:
        shutil.copy2(dataset_dir / name, sample_dir / name)

    (sample_dir / SAMPLE_NOTE_NAME).write_text(
        (
            "SPDMX sample for Mozilla Data Collective preview downloads.\n"
            f"Songs: {n_multitrack} multitrack + {n_single} single-track "
            f"(seed={seed}).\n"
            "Same layout as the full release; CSVs are truncated to this sample.\n"
        ),
        encoding="utf-8",
    )

    for _, row in tqdm(
        pick.iterrows(),
        total=len(pick),
        desc="sample:copy songs",
        unit="song",
    ):
        rel = _release_relpath(row["path"])
        src = dataset_dir / rel
        dst = sample_dir / rel
        if not src.is_dir():
            raise FileNotFoundError(f"Sample song directory missing: {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst)

    return pick


def part_archive_name(part: int) -> str:
    return f"spdmx-part{int(part)}.tar.gz"


def _index_member_names(source: Path, *, include_assignment: bool) -> list[str]:
    names = [
        *RELEASE_DOC_NAMES,
        f"{SPDMX_FILE_NAME}.csv",
        SPDMX_SONGS_FILE_NAME,
        CHUNKS_FILE_NAME,
    ]
    if include_assignment:
        names.append(CHUNK_ASSIGNMENT_FILE_NAME)
    missing = [name for name in names if not (source / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing release file(s): " + ", ".join(missing)
        )
    return names


def _part_payload_bytes(
    chunks: pd.DataFrame,
    chunk_ids: list[str],
    metadata_bytes: int,
) -> int:
    wanted = {normalize_chunk_id(chunk_id) for chunk_id in chunk_ids}
    column = chunks["chunk"].map(normalize_chunk_id)
    media = pd.to_numeric(chunks.loc[column.isin(wanted), "bytes"], errors="coerce")
    return int(media.fillna(0).sum()) + int(metadata_bytes)


def _stage_corpus_archives(
    source: Path,
    dest: Path,
    *,
    parts: int,
    max_part_bytes: int,
    pigz_level: int,
) -> list[Path]:
    if int(parts) == 1:
        full_path = dest / FULL_ARCHIVE_NAME
        print(f"writing {full_path} from {source} …", flush=True)
        stream_directory_to_tar_gz(
            source,
            full_path,
            pigz_level=pigz_level,
            desc=f"archive {FULL_ARCHIVE_NAME}",
        )
        print(f"  done ({full_path.stat().st_size} bytes)", flush=True)
        return [full_path]
    if int(parts) != DEFAULT_NUM_MDC_PARTS:
        raise ValueError(
            f"--parts must be 1 or {DEFAULT_NUM_MDC_PARTS}, got {parts}"
        )

    chunks = pd.read_csv(source / CHUNKS_FILE_NAME)
    if "chunk" not in chunks.columns or "bytes" not in chunks.columns:
        raise ValueError(f"{CHUNKS_FILE_NAME} missing chunk/bytes columns")
    chunk_ids = [normalize_chunk_id(chunk_id) for chunk_id in chunks["chunk"]]
    grouped = partition_chunk_ids(chunk_ids)
    assignment_on_disk = (source / CHUNK_ASSIGNMENT_FILE_NAME).is_file()
    bundled_assignment = bundled_chunk_assignment_path()
    if not assignment_on_disk and not bundled_assignment.is_file():
        raise FileNotFoundError(
            f"Missing {CHUNK_ASSIGNMENT_FILE_NAME} in {source} "
            f"and no bundled map at {bundled_assignment}"
        )

    parts_path = dest / PARTS_FILE_NAME
    parts_path.write_text(parts_document(), encoding="utf-8")
    extra_paths = [parts_path]
    if not assignment_on_disk:
        extra_paths.append(bundled_assignment)

    archives: list[Path] = []
    for part in (1, 2):
        part_chunks = grouped[part]
        members = _index_member_names(
            source, include_assignment=assignment_on_disk
        )
        for chunk_id in part_chunks:
            directory = chunk_dir_name(chunk_id)
            if not (source / directory).is_dir():
                raise FileNotFoundError(
                    f"Missing {directory} under {source} for part {part}"
                )
            members.append(directory)
        metadata_bytes = sum((source / name).stat().st_size for name in members if (source / name).is_file())
        metadata_bytes += sum(path.stat().st_size for path in extra_paths)
        payload = _part_payload_bytes(chunks, part_chunks, metadata_bytes)
        if payload > int(max_part_bytes):
            raise ValueError(
                f"part {part} is {payload} bytes of media plus metadata, "
                f"over the {int(max_part_bytes)} byte limit. "
                "Frozen songs are not moved to rebalance."
            )
        archive_path = dest / part_archive_name(part)
        print(
            f"writing {archive_path.name}: {len(part_chunks)} chunks, "
            f"~{payload} raw bytes …",
            flush=True,
        )
        stream_directory_to_tar_gz(
            source,
            archive_path,
            pigz_level=pigz_level,
            desc=f"archive {archive_path.name}",
            total_bytes=payload,
            members=members,
            extra_paths=extra_paths,
        )
        archives.append(archive_path)
        print(f"  done ({archive_path.stat().st_size} bytes)", flush=True)
    return archives


def stage_mdc_archives(
    *,
    dataset_dir: str | Path,
    stage_dir: str | Path,
    skip_full: bool = False,
    skip_sample: bool = False,
    sample_n_multitrack: int = DEFAULT_SAMPLE_N_MULTITRACK,
    sample_n_single: int = DEFAULT_SAMPLE_N_SINGLE,
    sample_seed: int = DEFAULT_SAMPLE_SEED,
    keep_sample_dir: bool = False,
    pigz_level: int = 1,
    parts: int = DEFAULT_NUM_MDC_PARTS,
    max_part_bytes: int = DEFAULT_MAX_PART_BYTES,
) -> list[Path]:
    """Build MDC archives. Returns paths of archives listed in ``SHA256SUMS``."""
    if skip_full and skip_sample:
        raise ValueError("nothing to do: both --skip-full and --skip-sample set")

    source = Path(dataset_dir)
    dest = Path(stage_dir)
    dest.mkdir(parents=True, exist_ok=True)
    _require_release_tree(source)

    archives: list[Path] = []

    if not skip_full:
        archives.extend(
            _stage_corpus_archives(
                source,
                dest,
                parts=int(parts),
                max_part_bytes=int(max_part_bytes),
                pigz_level=pigz_level,
            )
        )

    if not skip_sample:
        sample_dir = dest / SAMPLE_DIR_NAME
        print(
            f"building sample ({sample_n_multitrack} multitrack + "
            f"{sample_n_single} single, seed={sample_seed}) …",
            flush=True,
        )
        pick = build_sample_tree(
            dataset_dir=source,
            sample_dir=sample_dir,
            n_multitrack=sample_n_multitrack,
            n_single=sample_n_single,
            seed=sample_seed,
        )
        sample_path = dest / SAMPLE_ARCHIVE_NAME
        # tar -C sample_dir so archive root matches the full release layout.
        stream_directory_to_tar_gz(
            sample_dir,
            sample_path,
            pigz_level=pigz_level,
            desc=f"archive {SAMPLE_ARCHIVE_NAME}",
        )
        archives.append(sample_path)
        print(
            f"  sample archive {sample_path} "
            f"({len(pick)} songs, {sample_path.stat().st_size} bytes)",
            flush=True,
        )
        if not keep_sample_dir:
            shutil.rmtree(sample_dir)

    print("checksumming archives …", flush=True)
    _write_sha256sums(dest, archives)
    return archives


def main(argv=None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    dataset_dir = Path(
        args.dataset_dir
        if args.dataset_dir is not None
        else spdmx_dataset_dir(output_dir)
    )
    stage_dir = Path(
        args.stage_dir
        if args.stage_dir is not None
        else output_dir / DEFAULT_STAGE_DIR_NAME
    )
    try:
        archives = stage_mdc_archives(
            dataset_dir=dataset_dir,
            stage_dir=stage_dir,
            skip_full=bool(args.skip_full),
            skip_sample=bool(args.skip_sample),
            sample_n_multitrack=int(args.sample_n_multitrack),
            sample_n_single=int(args.sample_n_single),
            sample_seed=int(args.sample_seed),
            keep_sample_dir=bool(args.keep_sample_dir),
            pigz_level=int(args.pigz_level),
            parts=int(args.parts),
            max_part_bytes=int(args.max_part_bytes),
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(
        f"staged {len(archives)} archive(s) into {stage_dir} (from {dataset_dir})"
    )
    print(f"  {stage_dir / SHA256SUMS_FILE_NAME}")
    for path in archives:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
