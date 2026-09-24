"""Slice patch sweep variant stems into aligned 10s clips for swipe listening."""

from __future__ import annotations

import argparse
import multiprocessing
from pathlib import Path

import pandas as pd
import torch
import yaml
from tqdm import tqdm

from experiments.patch_sweep.clip_windows import (
    clip_id_for,
    clip_output_filename,
    find_content_rich_clips,
)
from experiments.patch_sweep.sweep import (
    MANIFEST_FILENAME,
    VARIANTS_DIR_NAME,
    default_source_dir,
    song_path_from_id,
)
from experiments.paths import DEFAULT_PROBE_STEMS
from experiments.probe_stems import active_probe_stems, load_probe_stems
from experiments.listening_shared.clips import DEFAULT_CLIP_SECONDS, clip_stem_waveform
from shared.config import CHUNK_SIZE, DATA_DIR_NAME, DEFAULT_AUDIO_FORMAT
from synthesis.audio import stem_filename, stem_is_valid, stem_path, write_audio

CLIP_MANIFEST_YAML = "clip_manifest.yaml"
CLIP_MANIFEST_CSV = "clip_manifest.csv"


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Build 10s listening clips from patch sweep variant outputs.",
    )
    parser.add_argument(
        "--sweep-dir",
        required=True,
        type=Path,
        help="Phase sweep output (e.g. output/phase1_archive_soundfonts).",
    )
    parser.add_argument(
        "--source-dir",
        default=None,
        type=Path,
        help="Basic ablation dir for reference stems (default: dev/ablations/basic).",
    )
    parser.add_argument(
        "--probe-stems",
        default=DEFAULT_PROBE_STEMS,
        type=Path,
    )
    parser.add_argument(
        "--categories",
        default=None,
        type=str,
        help="Comma-separated probe categories (default: all in sweep manifest).",
    )
    parser.add_argument(
        "--clips-per-stem",
        default=3,
        type=int,
    )
    parser.add_argument(
        "--clip-seconds",
        default=DEFAULT_CLIP_SECONDS,
        type=float,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild clips even when output files exist.",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        default=max(1, multiprocessing.cpu_count() // 2),
        type=int,
    )
    return parser.parse_args(args)


def _probe_filter(categories: str | None) -> set[str] | None:
    if not categories:
        return None
    return {part.strip() for part in categories.split(",") if part.strip()}


def _probe_clip_windows(
    args: tuple[dict, Path, int, float, bool],
) -> list[dict]:
    probe, source_dir, clips_per_stem, clip_seconds, show_hop_progress = args
    stem_id = probe["id"]
    track = int(probe["track"])
    song_path = song_path_from_id(source_dir, probe["song_id"])
    audio_format = None
    ref_path = None
    for ext in ("mp3", "flac"):
        candidate = stem_path(song_path, track, ext)
        if stem_is_valid(candidate):
            ref_path = candidate
            audio_format = ext
            break
    if ref_path is None:
        raise FileNotFoundError(f"Missing reference stem for probe {stem_id}: {song_path}")

    starts = find_content_rich_clips(
        ref_path,
        n_clips=clips_per_stem,
        clip_seconds=clip_seconds,
        show_progress=show_hop_progress,
    )
    if not starts:
        raise RuntimeError(f"No audible clip windows for probe stem: {stem_id}")

    return [
        {
            "clip_id": clip_id_for(stem_id, clip_index),
            "stem_id": stem_id,
            "category": probe.get("category"),
            "track": track,
            "clip_index": clip_index,
            "clip_start_seconds": float(start_seconds),
            "clip_seconds": float(clip_seconds),
            "reference_path": str(ref_path),
            "audio_format": audio_format,
        }
        for clip_index, start_seconds in enumerate(starts)
    ]


def build_clip_windows(
    *,
    probes: list[dict],
    source_dir: Path,
    clips_per_stem: int,
    clip_seconds: float,
    jobs: int = 1,
) -> list[dict]:
    tasks = [
        (probe, source_dir, clips_per_stem, clip_seconds, jobs <= 1)
        for probe in probes
    ]
    if jobs <= 1:
        clip_defs: list[dict] = []
        for task in tqdm(tasks, desc="Clip windows", unit="stem"):
            clip_defs.extend(_probe_clip_windows(task))
        return clip_defs

    with multiprocessing.Pool(processes=jobs) as pool:
        grouped = list(
            tqdm(
                pool.imap(_probe_clip_windows, tasks, chunksize=1),
                total=len(tasks),
                desc="Clip windows",
                unit="stem",
            )
        )
    return [clip_def for group in grouped for clip_def in group]


def _variant_stem_path(
    sweep_dir: Path,
    variant_id: str,
    song_id: str,
    track: int,
    audio_format: str,
) -> Path:
    return (
        sweep_dir
        / VARIANTS_DIR_NAME
        / variant_id
        / DATA_DIR_NAME
        / song_id
        / stem_filename(track, audio_format)
    )


def _clip_out_path(
    sweep_dir: Path,
    variant_id: str,
    song_id: str,
    track: int,
    clip_index: int,
    audio_format: str,
) -> Path:
    return (
        sweep_dir
        / "clips"
        / VARIANTS_DIR_NAME
        / variant_id
        / DATA_DIR_NAME
        / song_id
        / clip_output_filename(track, clip_index, audio_format)
    )


def _render_clip_task(args: tuple) -> dict | None:
    from synthesis.listening.catalog import song_id_from_path

    row, clip_def, sweep_dir, force = args
    variant_id = str(row["variant_id"])
    track = int(clip_def["track"])
    clip_index = int(clip_def["clip_index"])
    audio_format = str(row.get("audio_format") or clip_def["audio_format"] or DEFAULT_AUDIO_FORMAT)
    song_id = song_id_from_path(str(row["path"]))
    source_path = Path(str(row["out_path"]))
    if not source_path.is_file():
        source_path = _variant_stem_path(sweep_dir, variant_id, song_id, track, audio_format)
    out_path = _clip_out_path(sweep_dir, variant_id, song_id, track, clip_index, audio_format)
    if out_path.is_file() and not force:
        return {
            **row.to_dict(),
            "clip_id": clip_def["clip_id"],
            "clip_index": clip_index,
            "clip_start_seconds": clip_def["clip_start_seconds"],
            "clip_seconds": clip_def["clip_seconds"],
            "out_path": str(out_path.resolve()),
        }

    if not source_path.is_file():
        return None

    waveform = clip_stem_waveform(
        source_path,
        clip_seconds=float(clip_def["clip_seconds"]),
        start_seconds=float(clip_def["clip_start_seconds"]),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_audio(torch.from_numpy(waveform), out_path, audio_format)
    return {
        **row.to_dict(),
        "clip_id": clip_def["clip_id"],
        "clip_index": clip_index,
        "clip_start_seconds": clip_def["clip_start_seconds"],
        "clip_seconds": clip_def["clip_seconds"],
        "out_path": str(out_path.resolve()),
    }


def make_clips(
    *,
    sweep_dir: Path,
    source_dir: Path,
    probe_stems_path: Path,
    categories: set[str] | None,
    clips_per_stem: int,
    clip_seconds: float,
    force: bool,
    jobs: int,
) -> Path:
    sweep_dir = sweep_dir.resolve()
    source_dir = source_dir.resolve()
    manifest_path = sweep_dir / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing sweep manifest: {manifest_path}")

    manifest = pd.read_csv(manifest_path)
    all_probes = load_probe_stems(probe_stems_path)
    probes = active_probe_stems(all_probes)
    if categories:
        probes = [probe for probe in probes if probe.get("category") in categories]
        manifest = manifest[manifest["category"].isin(categories)]

    clip_defs = build_clip_windows(
        probes=probes,
        source_dir=source_dir,
        clips_per_stem=clips_per_stem,
        clip_seconds=clip_seconds,
        jobs=jobs,
    )

    yaml_path = sweep_dir / CLIP_MANIFEST_YAML
    with open(yaml_path, "w") as f:
        yaml.safe_dump({"clips": clip_defs}, f, sort_keys=False)

    clip_by_stem: dict[str, list[dict]] = {}
    for clip_def in clip_defs:
        clip_by_stem.setdefault(str(clip_def["stem_id"]), []).append(clip_def)

    tasks = []
    for _, row in manifest.iterrows():
        stem_id = str(row["stem_id"])
        for clip_def in clip_by_stem.get(stem_id, []):
            tasks.append((row, clip_def, sweep_dir, force))

    if jobs <= 1:
        rows = [
            result
            for task in tqdm(tasks, desc="Clip renders", unit="clip")
            if (result := _render_clip_task(task)) is not None
        ]
    else:
        with multiprocessing.Pool(processes=jobs) as pool:
            rows = [
                result
                for result in tqdm(
                    pool.imap(_render_clip_task, tasks, chunksize=CHUNK_SIZE),
                    total=len(tasks),
                    desc="Clip renders",
                    unit="clip",
                )
                if result is not None
            ]

    if not rows:
        raise RuntimeError("No clip rows produced.")

    new_rows = pd.DataFrame(rows)
    clip_csv = sweep_dir / CLIP_MANIFEST_CSV
    if clip_csv.is_file():
        existing = pd.read_csv(clip_csv)
        rendered_stems = set(new_rows["stem_id"].astype(str))
        kept = existing[~existing["stem_id"].astype(str).isin(rendered_stems)]
        clip_manifest = pd.concat([kept, new_rows], ignore_index=True)
    else:
        clip_manifest = new_rows
    clip_manifest.to_csv(clip_csv, index=False)
    print(f"Clip windows: {yaml_path}")
    print(f"Clip manifest: {clip_csv} ({len(clip_manifest)} rows)")
    return clip_csv


def main():
    args = parse_args()
    make_clips(
        sweep_dir=args.sweep_dir,
        source_dir=args.source_dir or default_source_dir(),
        probe_stems_path=args.probe_stems,
        categories=_probe_filter(args.categories),
        clips_per_stem=args.clips_per_stem,
        clip_seconds=args.clip_seconds,
        force=args.force,
        jobs=args.jobs,
    )


if __name__ == "__main__":
    main()
