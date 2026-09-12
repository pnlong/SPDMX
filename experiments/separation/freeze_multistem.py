"""Freeze train/val manifests for full multi-stem SPDMX (stem vs other).

Indexes ``songs.csv`` rows with ``n_stems_on_disk >= min_stems`` and a
resolvable mix + stem FLACs under the chunked release tree.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from experiments.separation.paths import SPDMX_ROOT, load_config, resolve_dev_dir
from experiments.separation.spdmx_io import resolve_spdmx_mix


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


def index_multistem_songs(
    spdmx_root: Path,
    *,
    min_stems: int = 2,
) -> pd.DataFrame:
    """Build one row per multi-stem song with resolvable audio."""
    from tqdm import tqdm

    songs_path = spdmx_root / "songs.csv"
    if not songs_path.is_file():
        raise FileNotFoundError(f"missing songs.csv under {spdmx_root}")

    songs = pd.read_csv(songs_path, low_memory=False)
    if "n_stems_on_disk" not in songs.columns:
        raise RuntimeError("songs.csv missing n_stems_on_disk")
    songs = songs[songs["n_stems_on_disk"].astype(int) >= int(min_stems)].copy()
    rows: list[dict] = []
    root = Path(spdmx_root)
    for _, row in tqdm(songs.iterrows(), total=len(songs), desc="index multistem"):
        song_id = str(row["song_id"])
        tracks = _parse_tracks(row.get("tracks"))
        if len(tracks) < int(min_stems):
            continue
        path_col = row.get("path")
        if path_col is None or (isinstance(path_col, float) and pd.isna(path_col)):
            continue
        song_dir = root / str(path_col).replace("\\", "/").lstrip("./")
        if not song_dir.is_dir():
            continue
        mix_path = song_dir / "mix.flac"
        if not mix_path.is_file():
            mix = resolve_spdmx_mix(root, row, song_id=song_id)
            if mix is None:
                continue
        ok_tracks: list[int] = []
        for track in tracks:
            if (song_dir / f"{int(track)}.flac").is_file():
                ok_tracks.append(int(track))
        if len(ok_tracks) < int(min_stems):
            continue
        dur = float(row["song_length"]) if "song_length" in row and pd.notna(row["song_length"]) else 0.0
        if dur <= 0:
            continue
        mix_col = row.get("mix")
        mix_rel = (
            str(mix_col)
            if mix_col is not None and not (isinstance(mix_col, float) and pd.isna(mix_col))
            else str(Path(str(path_col).replace("\\", "/").lstrip("./")) / "mix.flac")
        )
        rows.append(
            {
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
        )
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
    # Also stash under manifests/ for discoverability next to BDGP summary.
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
    args = parser.parse_args()

    cfg = load_config(args.config)
    spdmx_root = Path(args.spdmx_root or cfg.get("spdmx_root") or SPDMX_ROOT)
    out_dir = Path(args.out or (resolve_dev_dir(cfg) / "manifests"))
    min_stems = int(args.min_stems if args.min_stems is not None else cfg.get("min_stems", 2))
    val_fraction = float(
        args.val_fraction if args.val_fraction is not None else cfg.get("val_fraction", 0.02)
    )
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 43))

    print(f"indexing multi-stem songs under {spdmx_root} (min_stems={min_stems}) ...")
    index = index_multistem_songs(spdmx_root, min_stems=min_stems)
    print(f"indexed {len(index)} songs ({index['hours'].sum():.1f} h)")
    summary = freeze_multistem_manifests(
        index, out_dir, seed=seed, val_fraction=val_fraction,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
