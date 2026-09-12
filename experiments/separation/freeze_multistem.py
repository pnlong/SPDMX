"""Freeze train/val manifests for full multi-stem SPDMX (stem vs other).

Indexes ``songs.csv`` rows with ``n_stems_on_disk >= min_stems`` and a
resolvable mix + stem FLACs under the chunked release tree.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from experiments.separation.paths import SPDMX_ROOT, load_config, resolve_dev_dir


MULTISTEM_ARM = "multistem"


def _parse_tracks(raw: object) -> list[int]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    text = str(raw).strip()
    if not text:
        return []
    out: list[int] = []
    for part in text.replace(",", "|").split("|"):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def _index_one(args: tuple[str, dict, int]) -> dict | None:
    """Worker: validate one songs.csv row on disk."""
    root_s, row, min_stems = args
    root = Path(root_s)
    song_id = str(row["song_id"])
    tracks = _parse_tracks(row.get("tracks"))
    if len(tracks) < min_stems:
        return None
    path_col = row.get("path")
    if path_col is None or (isinstance(path_col, float) and pd.isna(path_col)):
        return None
    path_rel = str(path_col).replace("\\", "/").lstrip("./")
    song_dir = root / path_rel
    if not song_dir.is_dir():
        return None
    if not (song_dir / "mix.flac").is_file():
        return None
    ok_tracks = [int(t) for t in tracks if (song_dir / f"{int(t)}.flac").is_file()]
    if len(ok_tracks) < min_stems:
        return None
    dur_raw = row.get("song_length")
    dur = float(dur_raw) if dur_raw is not None and not (
        isinstance(dur_raw, float) and pd.isna(dur_raw)
    ) else 0.0
    if dur <= 0:
        return None
    mix_col = row.get("mix")
    mix_rel = (
        str(mix_col)
        if mix_col is not None and not (isinstance(mix_col, float) and pd.isna(mix_col))
        else f"{path_rel}/mix.flac"
    )
    return {
        "corpus": "spdmx",
        "split": "all",
        "song_id": song_id,
        "path": str(path_col),
        "mix": mix_rel,
        "tracks": "|".join(str(t) for t in ok_tracks),
        "n_stems": len(ok_tracks),
        "duration_sec": dur,
        "hours": dur / 3600.0,
    }


def index_multistem_songs(
    spdmx_root: Path,
    *,
    min_stems: int = 2,
    jobs: int = 1,
) -> pd.DataFrame:
    """Build one row per multi-stem song with resolvable audio."""
    songs_path = spdmx_root / "songs.csv"
    if not songs_path.is_file():
        raise FileNotFoundError(f"missing songs.csv under {spdmx_root}")

    songs = pd.read_csv(songs_path, low_memory=False)
    if "n_stems_on_disk" not in songs.columns:
        raise RuntimeError("songs.csv missing n_stems_on_disk")
    songs = songs[songs["n_stems_on_disk"].astype(int) >= int(min_stems)].copy()
    records = songs.to_dict(orient="records")
    root_s = str(Path(spdmx_root))
    min_stems_i = int(min_stems)
    payloads = [(root_s, rec, min_stems_i) for rec in records]

    n_jobs = max(1, int(jobs))
    rows: list[dict] = []
    if n_jobs <= 1:
        for payload in tqdm(payloads, desc="index multistem", unit="song"):
            hit = _index_one(payload)
            if hit is not None:
                rows.append(hit)
    else:
        chunksize = max(16, min(256, len(payloads) // (n_jobs * 8) or 16))
        with mp.Pool(processes=n_jobs) as pool:
            for hit in tqdm(
                pool.imap_unordered(_index_one, payloads, chunksize=chunksize),
                total=len(payloads),
                desc=f"index multistem (-j {n_jobs})",
                unit="song",
            ):
                if hit is not None:
                    rows.append(hit)

    if not rows:
        raise RuntimeError(f"no multi-stem songs found under {spdmx_root}")
    return pd.DataFrame(rows)


def freeze_multistem_manifests(
    index: pd.DataFrame,
    out_dir: Path,
    *,
    seed: int,
    val_fraction: float,
) -> dict:
    """Write ``multistem/train.csv`` and ``multistem/val.csv``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    arm_dir = out_dir / MULTISTEM_ARM
    arm_dir.mkdir(parents=True, exist_ok=True)

    shuffled = index.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_val = max(1, int(round(len(shuffled) * float(val_fraction))))
    n_val = min(n_val, max(1, len(shuffled) - 1))
    val_df = shuffled.iloc[:n_val].copy()
    train_df = shuffled.iloc[n_val:].copy()
    val_df["split"] = "val"
    train_df["split"] = "train"

    train_path = arm_dir / "train.csv"
    val_path = arm_dir / "val.csv"
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    summary = {
        "seed": seed,
        "val_fraction": val_fraction,
        "arm": MULTISTEM_ARM,
        "protocol": "stem_vs_other",
        "n_songs": int(len(shuffled)),
        "train_hours": float(train_df["hours"].sum()),
        "val_hours": float(val_df["hours"].sum()),
        "train": {"n": int(len(train_df)), "hours": float(train_df["hours"].sum()), "path": str(train_path)},
        "val": {"n": int(len(val_df)), "hours": float(val_df["hours"].sum()), "path": str(val_path)},
    }
    with open(arm_dir / "manifest_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "multistem_manifest_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "config_multistem.yaml",
    )
    parser.add_argument("--spdmx-root", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--min-stems", type=int, default=None)
    parser.add_argument("--val-fraction", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=None,
        help="Parallel workers for on-disk indexing (default: 16)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    spdmx_root = Path(args.spdmx_root or cfg.get("spdmx_root") or SPDMX_ROOT)
    out_dir = Path(args.out or (resolve_dev_dir(cfg) / "manifests"))
    min_stems = int(args.min_stems if args.min_stems is not None else cfg.get("min_stems", 2))
    val_fraction = float(
        args.val_fraction if args.val_fraction is not None else cfg.get("val_fraction", 0.02)
    )
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 43))
    jobs = int(args.jobs if args.jobs is not None else cfg.get("index_jobs", 16))

    print(
        f"indexing multi-stem songs under {spdmx_root} "
        f"(min_stems={min_stems}, -j {jobs}) ..."
    )
    index = index_multistem_songs(spdmx_root, min_stems=min_stems, jobs=jobs)
    print(f"indexed {len(index)} songs ({index['hours'].sum():.1f} h)")
    summary = freeze_multistem_manifests(
        index, out_dir, seed=seed, val_fraction=val_fraction,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
