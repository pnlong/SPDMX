"""Build 4-stem packs (bass/drums/guitar/piano) for Slakh and sPDMX."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from experiments.separation.audio_io import (
    audio_duration_seconds,
    load_mono,
    sum_stems,
    write_flac,
)
from experiments.separation.paths import (
    SLAKH_ROOT,
    SPDMX_ROOT,
    TARGETS,
    load_config,
    resolve_dev_dir,
)
from experiments.separation.spdmx_io import resolve_spdmx_stem
from experiments.separation.targets import gm_to_target, slakh_inst_class_to_target


def _eligible_from_targets(present: set[str], *, require_all: bool) -> bool:
    if require_all:
        return set(TARGETS).issubset(present)
    return bool(present & set(TARGETS))


def _write_pack(
    *,
    buckets: dict[str, list[Path]],
    dest: Path,
    sample_rate: int,
    corpus: str,
    split: str,
    song_id: str,
    out_root: Path,
    allow_missing: bool = False,
) -> dict | None:
    """Decode/sum/write one song pack. Returns index row or None on failure.

    When ``allow_missing`` is True, absent targets are written as silence (mix
    is still the sum of present stems). When False, all four targets are required.
    """
    dest = Path(dest)
    out_root = Path(out_root)
    if all((dest / f"{t}.flac").is_file() for t in TARGETS) and (dest / "mix.flac").is_file():
        dur = audio_duration_seconds(dest / "mix.flac")
        return {
            "corpus": corpus,
            "split": split,
            "song_id": song_id,
            "path": str(dest.relative_to(out_root)),
            "duration_sec": dur,
            "hours": dur / 3600.0,
        }

    present = [t for t in TARGETS if buckets.get(t)]
    if not present:
        return None
    if not allow_missing and any(not buckets.get(t) for t in TARGETS):
        return None

    dest.mkdir(parents=True, exist_ok=True)
    audio_by_target: dict[str, object] = {}
    try:
        # Decode present stems first to establish length; missing → zeros.
        for target in present:
            arrays = []
            for p in buckets[target]:
                a, _ = load_mono(Path(p), sample_rate=sample_rate)
                arrays.append(a)
            audio_by_target[target] = sum_stems(arrays)
        n = max(int(a.shape[0]) for a in audio_by_target.values())  # type: ignore[attr-defined]

        for target in TARGETS:
            if target in audio_by_target:
                a = audio_by_target[target]
                if a.shape[0] < n:  # type: ignore[attr-defined]
                    a = np.pad(a, (0, n - a.shape[0]))  # type: ignore[attr-defined]
                elif a.shape[0] > n:  # type: ignore[attr-defined]
                    a = a[:n]  # type: ignore[index]
                audio_by_target[target] = a
            else:
                audio_by_target[target] = np.zeros(n, dtype=np.float32)
            write_flac(dest / f"{target}.flac", audio_by_target[target], sample_rate)  # type: ignore[arg-type]
        mix = sum_stems([audio_by_target[t] for t in present])  # type: ignore[index]
        if mix.shape[0] < n:
            mix = np.pad(mix, (0, n - mix.shape[0]))
        elif mix.shape[0] > n:
            mix = mix[:n]
        write_flac(dest / "mix.flac", mix, sample_rate)
        dur = float(n) / float(sample_rate)
    except Exception as exc:  # noqa: BLE001
        print(f"skip {song_id}: {exc}")
        return None

    return {
        "corpus": corpus,
        "split": split,
        "song_id": song_id,
        "path": str(dest.relative_to(out_root)),
        "duration_sec": dur,
        "hours": dur / 3600.0,
    }


def _slakh_job(payload: dict) -> dict | None:
    """Worker: pack one Slakh track (picklable top-level)."""
    track_dir = Path(payload["track_dir"])
    split = payload["split"]
    out_root = Path(payload["out_root"])
    sample_rate = int(payload["sample_rate"])
    require_all = bool(payload["require_all"])

    meta_path = track_dir / "metadata.yaml"
    if not meta_path.is_file():
        return None
    with open(meta_path) as f:
        meta = yaml.safe_load(f) or {}
    stems_meta = meta.get("stems") or {}
    buckets: dict[str, list[Path]] = defaultdict(list)
    for sid, sm in stems_meta.items():
        if not sm.get("audio_rendered", True):
            continue
        stem_path = track_dir / "stems" / f"{sid}.flac"
        if not stem_path.is_file():
            stem_path = track_dir / "stems" / f"{sid}.wav"
        if not stem_path.is_file():
            continue
        target = slakh_inst_class_to_target(
            sm.get("inst_class"),
            is_drum=bool(sm.get("is_drum")),
        )
        if target is None:
            continue
        buckets[target].append(stem_path)

    if not _eligible_from_targets(set(buckets), require_all=require_all):
        return None

    dest = out_root / "slakh" / split / track_dir.name
    return _write_pack(
        buckets=buckets,
        dest=dest,
        sample_rate=sample_rate,
        corpus="slakh",
        split=split,
        song_id=track_dir.name,
        out_root=out_root,
        allow_missing=not require_all,
    )


def _spdmx_job(payload: dict) -> dict | None:
    """Worker: pack one sPDMX song from a pre-bucketed stem list."""
    buckets = {
        t: [Path(p) for p in paths]
        for t, paths in payload["buckets"].items()
    }
    return _write_pack(
        buckets=buckets,
        dest=Path(payload["dest"]),
        sample_rate=int(payload["sample_rate"]),
        corpus="spdmx",
        split="all",
        song_id=str(payload["song_id"]),
        out_root=Path(payload["out_root"]),
        allow_missing=bool(payload.get("allow_missing", False)),
    )


def _run_pool(jobs: list[dict], worker, *, jobs_n: int, desc: str) -> list[dict]:
    if not jobs:
        return []
    rows: list[dict] = []
    if jobs_n <= 1:
        for job in tqdm(jobs, desc=desc):
            row = worker(job)
            if row is not None:
                rows.append(row)
        return rows

    with ProcessPoolExecutor(max_workers=jobs_n) as pool:
        futures = [pool.submit(worker, job) for job in jobs]
        for fut in tqdm(as_completed(futures), total=len(futures), desc=desc):
            row = fut.result()
            if row is not None:
                rows.append(row)
    return rows


def prepare_slakh(
    slakh_root: Path,
    out_root: Path,
    *,
    sample_rate: int,
    require_all: bool,
    splits: tuple[str, ...] = ("train", "validation", "test"),
    max_songs: int | None = None,
    jobs: int = 1,
) -> list[dict]:
    """Remap Slakh redux tracks into out_root/slakh/{split}/{track_id}/…"""
    job_list: list[dict] = []
    for split in splits:
        split_dir = slakh_root / split
        if not split_dir.is_dir():
            continue
        track_dirs = sorted(p for p in split_dir.iterdir() if p.is_dir())
        for track_dir in track_dirs:
            job_list.append(
                {
                    "track_dir": str(track_dir),
                    "split": split,
                    "out_root": str(out_root),
                    "sample_rate": sample_rate,
                    "require_all": require_all,
                }
            )
            if max_songs is not None and len(job_list) >= max_songs:
                # Cap is on submitted tracks (eligibility applied in worker).
                break
        if max_songs is not None and len(job_list) >= max_songs:
            break

    rows = _run_pool(job_list, _slakh_job, jobs_n=jobs, desc="slakh")
    if max_songs is not None:
        rows = rows[:max_songs]
    return rows


def prepare_spdmx(
    spdmx_root: Path,
    out_root: Path,
    *,
    sample_rate: int,
    require_all: bool,
    min_targets: int = 1,
    csv_name: str = "stems.csv",
    max_songs: int | None = None,
    jobs: int = 1,
) -> list[dict]:
    """Remap sPDMX GM stems into out_root/spdmx/all/{song_id}/…

    Expects the **chunked release** layout by default (``stems.csv`` with
    ``path`` / ``chunk`` pointing at ``chunk_N/<song_id>/``). Flat ``SPDMX_dev``
    trees (``audio/`` / ``raw/``) work for lab use.

    When ``require_all`` is True (default), songs are taken from
    ``songs.csv`` ``subset:bdgp`` when that table exists; otherwise BDGP
    eligibility is recomputed from stem programs. When ``require_all`` is
    False, missing BDGP stems are packed as silence and ``min_targets``
    keeps only songs with at least that many of the four targets present.
    """
    from synthesis.build_songs_table import load_bdgp_song_ids

    csv_path = spdmx_root / csv_name
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)
    if "chunk" in df.columns:
        df["chunk"] = df["chunk"].map(lambda x: x if pd.isna(x) else str(int(x)))
    if "is_drum" in df.columns:
        df["is_drum"] = df["is_drum"].astype(str).str.lower().isin(("true", "1", "yes"))

    allow_missing = not require_all
    min_targets = max(1, int(min_targets))
    bdgp_ids = load_bdgp_song_ids(spdmx_root) if require_all else None
    if bdgp_ids is not None:
        before = int(df["song_id"].nunique())
        df = df[df["song_id"].astype(str).isin(bdgp_ids)]
        print(
            f"spdmx: using songs.csv subset:bdgp "
            f"({len(bdgp_ids)} songs; stems rows cover {df['song_id'].nunique()}/{before})"
        )
    elif require_all:
        print("spdmx: songs.csv subset:bdgp missing; recomputing BDGP from stems")

    job_list: list[dict] = []
    grouped = df.groupby("song_id", sort=False)
    n_songs = int(df["song_id"].nunique())
    for song_id, g in tqdm(grouped, desc="spdmx:index", total=n_songs):
        if max_songs is not None and len(job_list) >= max_songs:
            break
        buckets: dict[str, list[str]] = defaultdict(list)
        for _, r in g.iterrows():
            target = gm_to_target(int(r["program"]), bool(r["is_drum"]))
            if target is None:
                continue
            track = int(r["track"])
            stem_path = resolve_spdmx_stem(
                spdmx_root, r, song_id=str(song_id), track=track,
            )
            if stem_path is None:
                continue
            buckets[target].append(str(stem_path))

        present = set(buckets)
        if require_all:
            # Safety net if songs.csv was built with --no-check-files.
            if not set(TARGETS).issubset(present):
                continue
        else:
            if len(present) < min_targets:
                continue

        dest = out_root / "spdmx" / "all" / str(song_id)
        job_list.append(
            {
                "buckets": dict(buckets),
                "dest": str(dest),
                "sample_rate": sample_rate,
                "song_id": str(song_id),
                "out_root": str(out_root),
                "allow_missing": allow_missing,
            }
        )

    print(
        f"spdmx: queued {len(job_list)} songs "
        f"(require_all={require_all}, min_targets={min_targets}, "
        f"songs_csv_bdgp={'yes' if bdgp_ids is not None else 'no'})"
    )
    return _run_pool(job_list, _spdmx_job, jobs_n=jobs, desc="spdmx:pack")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--corpus", choices=("slakh", "spdmx", "both"), default="both")
    parser.add_argument("--slakh-root", type=Path, default=None)
    parser.add_argument("--spdmx-root", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--max-songs",
        type=int,
        default=None,
        help="Cap eligible songs packed per corpus (smoke tests)",
    )
    parser.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=max(1, min(8, (os.cpu_count() or 4))),
        help="Parallel workers for decode/encode (default: min(8, CPUs))",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    sample_rate = int(cfg.get("sample_rate", 44100))
    # Slakh keeps all-four by default; sPDMX can opt into partial packs.
    require_all_slakh = bool(cfg.get("require_all_targets", True))
    require_all_spdmx = bool(
        cfg.get("spdmx_require_all_targets", cfg.get("require_all_targets", True))
    )
    spdmx_min_targets = int(cfg.get("spdmx_min_targets", 1))
    out_root = args.out or (resolve_dev_dir(cfg) / "packs")
    out_root.mkdir(parents=True, exist_ok=True)
    jobs = max(1, int(args.jobs))
    print(f"prepare_stems: jobs={jobs}")

    all_rows: list[dict] = []
    if args.corpus in ("slakh", "both"):
        slakh_root = args.slakh_root or Path(cfg.get("slakh_root") or SLAKH_ROOT)
        all_rows.extend(
            prepare_slakh(
                slakh_root,
                out_root,
                sample_rate=sample_rate,
                require_all=require_all_slakh,
                max_songs=args.max_songs,
                jobs=jobs,
            )
        )
    if args.corpus in ("spdmx", "both"):
        spdmx_root = args.spdmx_root or Path(cfg.get("spdmx_root") or SPDMX_ROOT)
        all_rows.extend(
            prepare_spdmx(
                spdmx_root,
                out_root,
                sample_rate=sample_rate,
                require_all=require_all_spdmx,
                min_targets=spdmx_min_targets,
                max_songs=args.max_songs,
                jobs=jobs,
            )
        )

    index_path = out_root / "pack_index.csv"
    new_df = pd.DataFrame(all_rows)
    # When packing a single corpus, keep the other corpus's existing rows.
    if index_path.is_file() and args.corpus in ("slakh", "spdmx") and not new_df.empty:
        old = pd.read_csv(index_path)
        keep = old[old["corpus"] != args.corpus]
        new_df = pd.concat([keep, new_df], ignore_index=True)
    elif index_path.is_file() and args.corpus in ("slakh", "spdmx") and new_df.empty:
        new_df = pd.read_csv(index_path)

    new_df.to_csv(index_path, index=False)
    summary = {
        "n_rows": int(len(new_df)),
        "by_corpus": {
            c: float(new_df.query("corpus == @c")["hours"].sum())
            for c in sorted(new_df["corpus"].unique())
        },
        "index": str(index_path),
        "jobs": jobs,
        "require_all_slakh": require_all_slakh,
        "require_all_spdmx": require_all_spdmx,
        "spdmx_min_targets": spdmx_min_targets,
    }
    with open(out_root / "pack_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    # Required for ProcessPoolExecutor on some platforms.
    main()
