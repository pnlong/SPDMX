"""Slice ablation stems into aligned 10s clips for listening (windows from A1)."""

from __future__ import annotations

import argparse
import multiprocessing
from pathlib import Path

import pandas as pd
import torch
import yaml
from tqdm import tqdm

from experiments.patch_sweep.clip_windows import find_content_rich_clips
from experiments.listening_shared.clips import DEFAULT_CLIP_SECONDS, clip_stem_waveform
from shared.config import (
    DATA_DIR_NAME,
    DEFAULT_AUDIO_FORMAT,
    LISTENING_PREFER_SUMMABLE,
    LISTENING_SAMPLE_FILE_NAME,
    OUTPUT_DIR,
    STEMS_FILE_NAME,
)
from synthesis.audio import stem_is_valid, stem_path, write_audio
from synthesis.dataset import load_listening_sample
from synthesis.listening.catalog import (
    CONDITION_ORDER,
    _condition_dir,
    require_summable_condition_trees,
    song_id_from_path,
    summable_condition_name,
)
from synthesis.paths import ablations_root
from synthesis.patches import resolve_probe_category


CLIP_MANIFEST_CSV = "clip_manifest.csv"
CLIPS_DIR_NAME = "clips"


def _clip_condition_dirname(condition: str) -> str:
    """Directory name under clips/ (summable when LISTENING_PREFER_SUMMABLE)."""
    if LISTENING_PREFER_SUMMABLE:
        return summable_condition_name(condition)
    return condition


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Build aligned 10s listening clips from ablation stems (windows from basic/A1).",
    )
    parser.add_argument("-o", "--output_dir", default=OUTPUT_DIR, type=str)
    parser.add_argument(
        "--reference-mode",
        default="basic",
        type=str,
        help="Condition used to pick content-rich clip windows (default: basic).",
    )
    parser.add_argument(
        "--conditions",
        default=None,
        type=str,
        help="Comma-separated conditions to clip (default: all in CONDITION_ORDER).",
    )
    parser.add_argument("--clip-seconds", default=DEFAULT_CLIP_SECONDS, type=float)
    parser.add_argument(
        "--audio-format",
        default=DEFAULT_AUDIO_FORMAT,
        choices=["mp3", "flac"],
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
    return parser.parse_args(args=args)


def clips_root(output_dir: str) -> Path:
    return Path(ablations_root(output_dir)) / CLIPS_DIR_NAME


def _inventory_stems(output_dir: str, reference_dir: Path) -> list[dict]:
    sample_path = Path(ablations_root(output_dir)) / LISTENING_SAMPLE_FILE_NAME
    if sample_path.is_file():
        doc = load_listening_sample(sample_path)
        stems = list(doc.get("stems") or [])
        if stems:
            return stems

    stems_csv = reference_dir / f"{STEMS_FILE_NAME}.csv"
    if not stems_csv.is_file():
        raise FileNotFoundError(
            f"Need {sample_path} or {stems_csv} to build listening clips."
        )
    df = pd.read_csv(stems_csv)
    stems = []
    for _, row in df.iterrows():
        song_path = str(row["path"])
        stems.append({
            "song_id": song_id_from_path(song_path),
            "track": int(row["track"]),
            "original_track": (
                int(row["original_track"])
                if "original_track" in row.index and pd.notna(row.get("original_track"))
                else int(row["track"])
            ),
            "program": int(row["program"]) if pd.notna(row.get("program")) else 0,
            "is_drum": bool(row.get("is_drum", False)),
            "name": None if pd.isna(row.get("name")) else str(row.get("name")),
            "category": resolve_probe_category(
                program=int(row["program"]) if pd.notna(row.get("program")) else 0,
                is_drum=bool(row.get("is_drum", False)),
                track_name=None if pd.isna(row.get("name")) else str(row.get("name")),
            ),
        })
    return stems


def _window_task(task: tuple) -> dict | None:
    stem, ref_path, fmt, clip_seconds = task
    starts = find_content_rich_clips(
        Path(ref_path),
        n_clips=1,
        clip_seconds=clip_seconds,
        show_progress=False,
    )
    if not starts:
        return None
    return {
        **stem,
        "audio_format": fmt,
        "clip_start_seconds": float(starts[0]),
        "clip_seconds": float(clip_seconds),
        "reference_path": str(ref_path),
    }


def _pick_windows(
    stems: list[dict],
    reference_dir: Path,
    *,
    clip_seconds: float,
    audio_format: str,
    jobs: int,
) -> list[dict]:
    tasks = []
    for stem in stems:
        song_id = stem.get("song_id")
        if not song_id:
            continue
        track = int(stem["track"])
        ref_path = stem_path(reference_dir / DATA_DIR_NAME / song_id, track, audio_format)
        if not stem_is_valid(ref_path):
            alt = "flac" if audio_format == "mp3" else "mp3"
            ref_path = stem_path(reference_dir / DATA_DIR_NAME / song_id, track, alt)
            if not stem_is_valid(ref_path):
                continue
            fmt = alt
        else:
            fmt = audio_format
        tasks.append((stem, str(ref_path), fmt, clip_seconds))

    if jobs <= 1:
        windows = []
        for task in tqdm(tasks, desc="Clip windows", unit="stem"):
            row = _window_task(task)
            if row:
                windows.append(row)
        return windows

    with multiprocessing.Pool(processes=jobs) as pool:
        rows = list(
            tqdm(
                pool.imap(_window_task, tasks, chunksize=1),
                total=len(tasks),
                desc="Clip windows",
                unit="stem",
            )
        )
    return [row for row in rows if row]


def _render_clip_task(args: tuple) -> dict | None:
    window, condition, ablations_dir, clips_dir, force = args
    song_id = window["song_id"]
    track = int(window["track"])
    audio_format = str(window["audio_format"])
    src_root = _condition_dir(Path(ablations_dir), condition)
    out_root = Path(clips_dir) / _clip_condition_dirname(condition)
    src = stem_path(
        src_root / DATA_DIR_NAME / song_id,
        track,
        audio_format,
    )
    out = stem_path(
        out_root / DATA_DIR_NAME / song_id,
        track,
        audio_format,
    )
    if out.is_file() and not force:
        return {
            **window,
            "condition": condition,
            "out_path": str(out),
        }
    if not stem_is_valid(src):
        return None
    waveform = clip_stem_waveform(
        src,
        clip_seconds=float(window["clip_seconds"]),
        start_seconds=float(window["clip_start_seconds"]),
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    write_audio(torch.from_numpy(waveform), out, audio_format)
    return {
        **window,
        "condition": condition,
        "out_path": str(out),
    }


def make_clips(
    *,
    output_dir: str,
    reference_mode: str = "basic",
    conditions: list[str] | None = None,
    clip_seconds: float = DEFAULT_CLIP_SECONDS,
    audio_format: str = DEFAULT_AUDIO_FORMAT,
    force: bool = False,
    jobs: int = 1,
) -> Path:
    ablations_dir = Path(ablations_root(output_dir))
    require_summable_condition_trees(ablations_dir)
    reference_dir = _condition_dir(ablations_dir, reference_mode)
    if not (reference_dir / f"{DATA_DIR_NAME}.csv").is_file():
        raise FileNotFoundError(
            f"Reference ablation missing data.csv: {reference_dir}\n"
            f"Run mix for summable stems, or synthesis for raw:\n"
            f"  uv run python -m synthesis.mix --render-mode {reference_mode} --no-overwrite -j 8\n"
            f"  uv run python -m synthesis.synthesize --render-mode {reference_mode}"
        )

    conditions = conditions or [c for c in CONDITION_ORDER]
    stems = _inventory_stems(output_dir, reference_dir)
    windows = _pick_windows(
        stems,
        reference_dir,
        clip_seconds=clip_seconds,
        audio_format=audio_format,
        jobs=jobs,
    )
    if not windows:
        raise RuntimeError("No audible 10s clip windows found in reference stems.")

    out_root = clips_root(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Persist window choices once for reproducibility.
    with open(out_root / "clip_windows.yaml", "w") as f:
        yaml.safe_dump(
            {
                "reference_mode": reference_mode,
                "clip_seconds": clip_seconds,
                "prefer_summable": LISTENING_PREFER_SUMMABLE,
                "windows": windows,
            },
            f,
            sort_keys=False,
            default_flow_style=False,
        )

    tasks = [
        (window, condition, ablations_dir, out_root, force)
        for window in windows
        for condition in conditions
        if (_condition_dir(ablations_dir, condition) / f"{DATA_DIR_NAME}.csv").is_file()
    ]
    manifest_rows: list[dict] = []
    if jobs <= 1:
        for task in tqdm(tasks, desc="Writing clips", unit="clip"):
            row = _render_clip_task(task)
            if row:
                manifest_rows.append(row)
    else:
        with multiprocessing.Pool(processes=jobs) as pool:
            for row in tqdm(
                pool.imap(_render_clip_task, tasks, chunksize=1),
                total=len(tasks),
                desc="Writing clips",
                unit="clip",
            ):
                if row:
                    manifest_rows.append(row)

    manifest_path = out_root / CLIP_MANIFEST_CSV
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

    # Mini catalogs per condition so AblationCatalog can load clips as reference.
    _write_clip_condition_tables(out_root, windows, conditions, ablations_dir)
    print(f"Wrote {len(manifest_rows)} clips under {out_root}")
    return manifest_path


def _write_clip_condition_tables(
    clips_dir: Path,
    windows: list[dict],
    conditions: list[str],
    ablations_dir: Path,
) -> None:
    # Use reference condition songs table as template; rewrite paths into clips tree.
    ref_condition = next(
        (
            c for c in conditions
            if (_condition_dir(ablations_dir, c) / f"{DATA_DIR_NAME}.csv").is_file()
        ),
        None,
    )
    if ref_condition is None:
        return
    songs = pd.read_csv(
        _condition_dir(ablations_dir, ref_condition) / f"{DATA_DIR_NAME}.csv"
    )
    song_ids = {w["song_id"] for w in windows}
    keep_paths = []
    path_map = {}
    for _, row in songs.iterrows():
        try:
            sid = song_id_from_path(str(row["path"]))
        except ValueError:
            continue
        if sid in song_ids:
            keep_paths.append(str(row["path"]))
            path_map[str(row["path"])] = sid

    songs = songs[songs["path"].astype(str).isin(keep_paths)].copy()

    stem_rows = []
    for w in windows:
        stem_rows.append({
            "path": None,  # filled per condition
            "track": w["track"],
            "original_track": w.get("original_track", w["track"]),
            "program": w.get("program", 0),
            "is_drum": w.get("is_drum", False),
            "name": w.get("name"),
            "has_lyrics": False,
            "song_id": w["song_id"],
            "category": w.get("category"),
            "clip_start_seconds": w["clip_start_seconds"],
            "clip_seconds": w["clip_seconds"],
        })

    for condition in conditions:
        if not (_condition_dir(ablations_dir, condition) / f"{DATA_DIR_NAME}.csv").is_file():
            continue
        cond_dir = clips_dir / _clip_condition_dirname(condition)
        cond_dir.mkdir(parents=True, exist_ok=True)
        songs_out = songs.copy()
        songs_out["path"] = songs_out["path"].map(
            lambda p: str(cond_dir / DATA_DIR_NAME / path_map[str(p)])
        )
        # Clip duration is fixed.
        if "song_length.seconds" in songs_out.columns:
            songs_out["song_length.seconds"] = windows[0]["clip_seconds"] if windows else 10.0
        songs_out.to_csv(cond_dir / f"{DATA_DIR_NAME}.csv", index=False)

        stems_out = []
        for stem in stem_rows:
            stems_out.append({
                "path": str(cond_dir / DATA_DIR_NAME / stem["song_id"]),
                "track": stem["track"],
                "original_track": stem.get("original_track", stem["track"]),
                "program": stem["program"],
                "is_drum": stem["is_drum"],
                "name": stem["name"],
                "has_lyrics": False,
            })
        pd.DataFrame(stems_out).to_csv(cond_dir / f"{STEMS_FILE_NAME}.csv", index=False)


def main(args=None):
    ns = parse_args(args)
    conditions = None
    if ns.conditions:
        conditions = [c.strip() for c in ns.conditions.split(",") if c.strip()]
    make_clips(
        output_dir=ns.output_dir,
        reference_mode=ns.reference_mode,
        conditions=conditions,
        clip_seconds=ns.clip_seconds,
        audio_format=ns.audio_format,
        force=ns.force,
        jobs=ns.jobs,
    )


if __name__ == "__main__":
    main()
