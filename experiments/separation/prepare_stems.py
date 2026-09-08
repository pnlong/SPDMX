"""Build 4-stem packs (bass/drums/guitar/piano) for Slakh and sPDMX."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

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


def prepare_slakh(
    slakh_root: Path,
    out_root: Path,
    *,
    sample_rate: int,
    require_all: bool,
    splits: tuple[str, ...] = ("train", "validation", "test"),
    max_songs: int | None = None,
) -> list[dict]:
    """Remap Slakh redux tracks into out_root/slakh/{split}/{track_id}/…"""
    rows: list[dict] = []
    n_done = 0
    for split in splits:
        split_dir = slakh_root / split
        if not split_dir.is_dir():
            continue
        track_dirs = sorted(p for p in split_dir.iterdir() if p.is_dir())
        for track_dir in tqdm(track_dirs, desc=f"slakh:{split}"):
            if max_songs is not None and n_done >= max_songs:
                return rows
            meta_path = track_dir / "metadata.yaml"
            if not meta_path.is_file():
                continue
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
                continue

            dest = out_root / "slakh" / split / track_dir.name
            dest.mkdir(parents=True, exist_ok=True)
            packed: dict[str, Path] = {}
            audio_by_target: dict[str, object] = {}
            for target in TARGETS:
                paths = buckets.get(target) or []
                if not paths:
                    continue
                arrays = []
                sr_used = sample_rate
                for p in paths:
                    a, sr_used = load_mono(p, sample_rate=sample_rate)
                    arrays.append(a)
                merged = sum_stems(arrays)
                out_p = dest / f"{target}.flac"
                write_flac(out_p, merged, sr_used)
                packed[target] = out_p
                audio_by_target[target] = merged

            if set(TARGETS) - set(packed):
                continue
            mix = sum_stems([audio_by_target[t] for t in TARGETS])  # type: ignore[index]
            write_flac(dest / "mix.flac", mix, sample_rate)
            dur = float(mix.shape[0]) / float(sample_rate)
            rows.append(
                {
                    "corpus": "slakh",
                    "split": split,
                    "song_id": track_dir.name,
                    "path": str(dest.relative_to(out_root)),
                    "duration_sec": dur,
                    "hours": dur / 3600.0,
                }
            )
            n_done += 1
    return rows


def prepare_spdmx(
    spdmx_root: Path,
    out_root: Path,
    *,
    sample_rate: int,
    require_all: bool,
    csv_name: str = "SPDMX.csv",
    max_songs: int | None = None,
) -> list[dict]:
    """Remap sPDMX GM stems into out_root/spdmx/all/{song_id}/…

    Expects the **chunked release** layout by default (``SPDMX.csv`` with
    ``path`` / ``chunk`` pointing at ``chunk_N/audio/…``). Flat ``SPDMX_dev``
    trees still work as a fallback.
    """
    csv_path = spdmx_root / csv_name
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)
    if "chunk" in df.columns:
        # Preserve unpadded ids if written as strings; ints are fine too.
        df["chunk"] = df["chunk"].map(lambda x: x if pd.isna(x) else str(int(x)))
    if "is_drum" in df.columns:
        df["is_drum"] = df["is_drum"].astype(str).str.lower().isin(("true", "1", "yes"))

    rows: list[dict] = []
    grouped = df.groupby("song_id", sort=False)
    n_songs = int(df["song_id"].nunique())
    n_done = 0
    for song_id, g in tqdm(grouped, desc="spdmx", total=n_songs):
        if max_songs is not None and n_done >= max_songs:
            break
        buckets: dict[str, list[Path]] = defaultdict(list)
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
            buckets[target].append(stem_path)

        if not _eligible_from_targets(set(buckets), require_all=require_all):
            continue

        if any(not buckets.get(t) for t in TARGETS):
            continue

        dest = out_root / "spdmx" / "all" / str(song_id)
        if all((dest / f"{t}.flac").is_file() for t in TARGETS) and (dest / "mix.flac").is_file():
            dur = audio_duration_seconds(dest / "mix.flac")
            rows.append(
                {
                    "corpus": "spdmx",
                    "split": "all",
                    "song_id": str(song_id),
                    "path": str(dest.relative_to(out_root)),
                    "duration_sec": dur,
                    "hours": dur / 3600.0,
                }
            )
            n_done += 1
            continue

        dest.mkdir(parents=True, exist_ok=True)
        audio_by_target: dict[str, object] = {}
        try:
            for target in TARGETS:
                arrays = []
                for p in buckets[target]:
                    a, _ = load_mono(p, sample_rate=sample_rate)
                    arrays.append(a)
                merged = sum_stems(arrays)
                write_flac(dest / f"{target}.flac", merged, sample_rate)
                audio_by_target[target] = merged
            mix = sum_stems([audio_by_target[t] for t in TARGETS])  # type: ignore[index]
            write_flac(dest / "mix.flac", mix, sample_rate)
            dur = float(mix.shape[0]) / float(sample_rate)
        except Exception as exc:  # noqa: BLE001
            print(f"skip {song_id}: {exc}")
            continue

        rows.append(
            {
                "corpus": "spdmx",
                "split": "all",
                "song_id": str(song_id),
                "path": str(dest.relative_to(out_root)),
                "duration_sec": dur,
                "hours": dur / 3600.0,
            }
        )
        n_done += 1
    return rows


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
    args = parser.parse_args()

    cfg = load_config(args.config)
    sample_rate = int(cfg.get("sample_rate", 44100))
    require_all = bool(cfg.get("require_all_targets", True))
    out_root = args.out or (resolve_dev_dir(cfg) / "packs")
    out_root.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    if args.corpus in ("slakh", "both"):
        slakh_root = args.slakh_root or Path(cfg.get("slakh_root") or SLAKH_ROOT)
        all_rows.extend(
            prepare_slakh(
                slakh_root,
                out_root,
                sample_rate=sample_rate,
                require_all=require_all,
                max_songs=args.max_songs,
            )
        )
    if args.corpus in ("spdmx", "both"):
        spdmx_root = args.spdmx_root or Path(cfg.get("spdmx_root") or SPDMX_ROOT)
        all_rows.extend(
            prepare_spdmx(
                spdmx_root,
                out_root,
                sample_rate=sample_rate,
                require_all=require_all,
                max_songs=args.max_songs,
            )
        )

    index_path = out_root / "pack_index.csv"
    pd.DataFrame(all_rows).to_csv(index_path, index=False)
    summary = {
        "n_rows": len(all_rows),
        "by_corpus": {
            c: float(pd.DataFrame(all_rows).query("corpus == @c")["hours"].sum())
            for c in sorted({r["corpus"] for r in all_rows})
        },
        "index": str(index_path),
    }
    with open(out_root / "pack_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
