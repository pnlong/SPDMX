"""Render full-song mixes under ``SPDMX_dev/mix/<song_id>.flac`` via ffmpeg.

After summable stems exist in ``audio/<song_id>/*.flac``, this pass sums them
with ffmpeg ``amix`` (``normalize=0`` = raw linear sum) and writes a mono
FLAC beside the dense MIDI tree (same channel layout as the stems).

Resumable: skips songs whose ``mix/<song_id>.flac`` already exists (size > 0)
unless ``--force`` / ``synthesis.final --reset``.

Example::

    uv run python -m synthesis.render_mixes -j 32
    uv run python -m synthesis.final --only-pass song_mix -j 32
"""

from __future__ import annotations

import argparse
import multiprocessing
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from analysis.corrected_midi import TRACK_MAP_FILE_NAME
from shared.config import (
    FLAC_AUDIO_FORMAT,
    OUTPUT_DIR,
    SPDMX_AUDIO_DIR_NAME,
    SPDMX_FILE_NAME,
    SPDMX_MIX_DIR_NAME,
)
from synthesis.paths import spdmx_audio_dir, spdmx_dev_dir, spdmx_mix_dir


def parse_args(args=None, namespace=None):
    parser = argparse.ArgumentParser(
        description=(
            "Sum SPDMX_dev/audio stems into SPDMX_dev/mix/<song_id>.flac "
            "(ffmpeg amix, normalize=0, mono)."
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
        help="Flat production tree (default: {output_dir}/SPDMX_dev).",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=8,
        help="Parallel ffmpeg workers (default: 8).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-render mixes even when the destination FLAC already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List songs that would be rendered without writing.",
    )
    return parser.parse_args(args=args, namespace=namespace)


def _stem_flac_paths(song_audio_dir: Path) -> list[Path]:
    """Dense track stems ``0.flac``, ``1.flac``, … (numeric basename only)."""
    if not song_audio_dir.is_dir():
        return []
    stems: list[tuple[int, Path]] = []
    for path in song_audio_dir.iterdir():
        if not path.is_file() or path.suffix.lower() != f".{FLAC_AUDIO_FORMAT}":
            continue
        try:
            track = int(path.stem)
        except ValueError:
            continue
        stems.append((track, path))
    stems.sort(key=lambda item: item[0])
    return [path for _, path in stems]


def _mix_ready(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def ffmpeg_sum_stems(
    stem_paths: list[Path],
    dest: Path,
    *,
    force: bool = False,
) -> str:
    """Write mono FLAC mix via ffmpeg ``amix`` (raw sum). Return status tag."""
    if not stem_paths:
        return "skip_no_stems"
    if not force and _mix_ready(dest):
        return "skip_exists"

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise FileNotFoundError("ffmpeg not found on PATH")

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    if tmp.exists():
        tmp.unlink()

    n = len(stem_paths)
    cmd: list[str] = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error"]
    for path in stem_paths:
        cmd.extend(["-i", str(path)])

    # Raw linear sum (normalize=0). Keep mono to match stem channel layout.
    inputs = "".join(f"[{i}:a]" for i in range(n))
    filter_complex = (
        f"{inputs}amix=inputs={n}:duration=longest:normalize=0,"
        f"aformat=channel_layouts=mono:sample_fmts=s16[aout]"
    )
    cmd.extend(
        [
            "-filter_complex",
            filter_complex,
            "-map",
            "[aout]",
            "-c:a",
            "flac",
            "-f",
            "flac",
            str(tmp),
        ]
    )
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not tmp.is_file() or tmp.stat().st_size <= 0:
        if tmp.exists():
            tmp.unlink()
        err = (proc.stderr or proc.stdout or "").strip() or f"exit {proc.returncode}"
        raise RuntimeError(f"ffmpeg failed for {dest}: {err}")
    tmp.replace(dest)
    return "wrote"


def song_ids_from_stems_csv(dataset_dir: Path) -> list[str]:
    csv_path = dataset_dir / TRACK_MAP_FILE_NAME
    if not csv_path.is_file():
        csv_path = dataset_dir / f"{SPDMX_FILE_NAME}.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Missing {dataset_dir / TRACK_MAP_FILE_NAME}; "
            "run prepare_synthesis / synthesis.final first."
        )
    table = pd.read_csv(csv_path, usecols=["song_id"])
    return sorted(table["song_id"].astype(str).unique())


def update_stems_csv_mix_column(dataset_dir: Path) -> None:
    """Set flat ``mix`` column to ``./mix/<song_id>.flac`` when present."""
    csv_path = dataset_dir / f"{SPDMX_FILE_NAME}.csv"
    if not csv_path.is_file():
        return
    table = pd.read_csv(csv_path)
    if "song_id" not in table.columns:
        return
    table["mix"] = table["song_id"].astype(str).map(
        lambda sid: f"./{SPDMX_MIX_DIR_NAME}/{sid}.flac"
    )
    table.to_csv(csv_path, index=False)


def _render_one_song_mix(payload: dict) -> str:
    """Picklable process-pool worker: render one song mix; return status tag."""
    song_id = str(payload["song_id"])
    audio_root = Path(payload["audio_root"])
    mix_root = Path(payload["mix_root"])
    force = bool(payload["force"])
    dry_run = bool(payload["dry_run"])
    dest = mix_root / f"{song_id}.flac"
    # Resume first: avoid NFS stem-dir listing when the mix already exists.
    if not force and _mix_ready(dest):
        return "skip_exists"
    stems = _stem_flac_paths(audio_root / song_id)
    if dry_run:
        if not stems:
            return "skip_no_stems"
        return "wrote"
    try:
        return ffmpeg_sum_stems(stems, dest, force=force)
    except Exception as exc:  # noqa: BLE001
        print(f"error: mix {song_id}: {exc}", file=sys.stderr)
        return "error"


def render_dataset_mixes(
    dataset_dir: str | Path,
    *,
    jobs: int = 8,
    force: bool = False,
    dry_run: bool = False,
    update_csv: bool = True,
) -> dict[str, int]:
    """Render mixes for every song in ``stems.csv``. Returns status counts.

    Resumable: existing non-empty ``mix/<song_id>.flac`` files are skipped unless
    ``force`` is set. Uses a ``multiprocessing.Pool`` when ``jobs > 1``.
    """
    root = Path(dataset_dir)
    audio_root = root / SPDMX_AUDIO_DIR_NAME
    mix_root = root / SPDMX_MIX_DIR_NAME
    if not audio_root.is_dir():
        raise FileNotFoundError(f"Missing flat audio tree: {audio_root}")

    song_ids = song_ids_from_stems_csv(root)
    counts = {"wrote": 0, "skip_exists": 0, "skip_no_stems": 0, "error": 0}

    # Parent-process resume filter: only enqueue songs that still need work.
    pending_ids: list[str] = []
    if force:
        pending_ids = list(song_ids)
    else:
        for song_id in tqdm(song_ids, desc="song_mix resume scan"):
            if _mix_ready(mix_root / f"{song_id}.flac"):
                counts["skip_exists"] += 1
            else:
                pending_ids.append(song_id)
        print(
            f"song_mix: {counts['skip_exists']} already done, "
            f"{len(pending_ids)} remaining",
            flush=True,
        )

    payloads = [
        {
            "song_id": song_id,
            "audio_root": str(audio_root),
            "mix_root": str(mix_root),
            "force": force,
            "dry_run": dry_run,
        }
        for song_id in pending_ids
    ]

    n_jobs = max(1, int(jobs))
    results: list[str] = []
    if not payloads:
        pass
    elif n_jobs <= 1 or len(payloads) <= 1:
        results = [
            _render_one_song_mix(payload)
            for payload in tqdm(payloads, desc="render mixes")
        ]
    else:
        label = f"render mixes (-j {n_jobs})"
        chunksize = max(1, min(8, len(payloads) // (n_jobs * 8) or 1))
        with multiprocessing.Pool(processes=n_jobs) as pool:
            for status in tqdm(
                pool.imap_unordered(_render_one_song_mix, payloads, chunksize=chunksize),
                total=len(payloads),
                desc=label,
            ):
                results.append(status)

    for status in results:
        # skip_exists from workers is rare after the parent filter; still count.
        counts[str(status)] = counts.get(str(status), 0) + 1

    if update_csv and not dry_run:
        update_stems_csv_mix_column(root)
    return counts


def main(argv=None) -> int:
    args = parse_args(argv)
    dataset_dir = Path(
        args.dataset_dir
        if args.dataset_dir is not None
        else spdmx_dev_dir(args.output_dir)
    )
    # Touch helpers so imports stay accurate for callers.
    _ = (spdmx_audio_dir(args.output_dir), spdmx_mix_dir(args.output_dir))
    try:
        counts = render_dataset_mixes(
            dataset_dir,
            jobs=args.jobs,
            force=args.force,
            dry_run=args.dry_run,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    mode = "dry-run" if args.dry_run else "done"
    print(
        f"{mode}: {dataset_dir / SPDMX_MIX_DIR_NAME}/ "
        f"wrote={counts.get('wrote', 0)} "
        f"skip_exists={counts.get('skip_exists', 0)} "
        f"skip_no_stems={counts.get('skip_no_stems', 0)} "
        f"error={counts.get('error', 0)}"
    )
    return 1 if counts.get("error", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
