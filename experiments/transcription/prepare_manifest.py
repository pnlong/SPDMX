"""Build song-level manifests for Slakh-redux and SPDMX multitrack AMT.

Each row: corpus, split, song_id, mix_path, midi_path, n_stems, duration_sec.
SPDMX uses packaged chunk layout (mix.flac + mix.mid). Slakh uses redux splits
only by default (no omitted/), matching Slakh transcription guidance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import soundfile as sf
import yaml

from experiments.transcription.paths import (
    REPO_ROOT,
    SLAKH_ROOT,
    SPDMX_ROOT,
    load_config,
    resolve_dev_dir,
)


def _split_songs(song_ids: list[str], *, seed: int, val_frac: float, test_frac: float):
    import random

    ids = list(song_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_test = max(1, int(round(n * test_frac))) if n > 2 else 0
    n_val = max(1, int(round(n * val_frac))) if n > 2 else 0
    test = ids[:n_test]
    val = ids[n_test : n_test + n_val]
    train = ids[n_test + n_val :]
    return {"train": train, "validation": val, "test": test}


def index_spdmx(
    spdmx_root: Path,
    *,
    min_stems: int = 2,
    seed: int = 43,
    val_frac: float = 0.05,
    test_frac: float = 0.05,
) -> pd.DataFrame:
    songs_csv = spdmx_root / "songs.csv"
    if not songs_csv.is_file():
        raise FileNotFoundError(f"missing {songs_csv}")
    songs = pd.read_csv(songs_csv)
    songs["n_tracks"] = pd.to_numeric(songs["n_tracks"], errors="coerce")
    songs["song_length"] = pd.to_numeric(songs.get("song_length"), errors="coerce")
    songs = songs[songs["n_tracks"] >= int(min_stems)].copy()
    if songs.empty:
        raise RuntimeError(f"no SPDMX songs with n_tracks>={min_stems}")

    splits = _split_songs(
        songs["song_id"].astype(str).tolist(),
        seed=seed,
        val_frac=val_frac,
        test_frac=test_frac,
    )
    split_of = {sid: sp for sp, ids in splits.items() for sid in ids}

    rows = []
    for _, row in songs.iterrows():
        song_id = str(row["song_id"])
        # Prefer table paths (no NFS probes). Fall back to conventional layout strings.
        mix = ""
        midi = ""
        if "mix" in row and pd.notna(row["mix"]):
            mix = str(row["mix"])
        elif pd.notna(row.get("chunk")):
            mix = str(spdmx_root / f"chunk_{int(row['chunk'])}" / song_id / "mix.flac")
        if "mid" in row and pd.notna(row["mid"]):
            midi = str(row["mid"])
        elif pd.notna(row.get("chunk")):
            midi = str(spdmx_root / f"chunk_{int(row['chunk'])}" / song_id / "mix.mid")
        dur = float(row["song_length"]) if pd.notna(row.get("song_length")) else None
        rows.append(
            {
                "corpus": "spdmx",
                "split": split_of.get(song_id, "train"),
                "song_id": song_id,
                "mix_path": mix,
                "midi_path": midi,
                "n_stems": int(row["n_tracks"]),
                "duration_sec": dur,
            }
        )
    return pd.DataFrame(rows)


def index_slakh(
    slakh_root: Path,
    *,
    include_omitted: bool = False,
    cache_path: Path | None = None,
) -> pd.DataFrame:
    """Index Slakh tracks; reuse ``slakh_track_hours.json`` cache when present."""
    from analysis.song_hours_by_stems import PAPER_DATA

    cache = cache_path or (PAPER_DATA / "slakh_track_hours.json")
    if cache.is_file():
        cached = json.loads(cache.read_text(encoding="utf-8"))
        rows = []
        for row in cached:
            split = row.get("split", "train")
            if split == "omitted" and not include_omitted:
                continue
            track = slakh_root / split / row["track"]
            mix = track / "mix.flac"
            midi = track / "all_src.mid"
            rows.append(
                {
                    "corpus": "slakh",
                    "split": "train" if split == "omitted" else split,
                    "song_id": row["track"],
                    "mix_path": str(mix),
                    "midi_path": str(midi),
                    "n_stems": int(row["n_stems"]),
                    "duration_sec": float(row["duration_sec"]),
                }
            )
        return pd.DataFrame(rows)

    splits = ["train", "validation", "test"]
    if include_omitted:
        splits.append("omitted")
    rows = []
    for split in splits:
        split_dir = slakh_root / split
        if not split_dir.is_dir():
            continue
        for track in sorted(split_dir.iterdir()):
            if not track.is_dir():
                continue
            mix = track / "mix.flac"
            midi = track / "all_src.mid"
            meta_path = track / "metadata.yaml"
            if not mix.is_file():
                continue
            n_stems = 0
            if meta_path.is_file():
                meta = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
                stems = meta.get("stems") or {}
                for s in stems.values():
                    if isinstance(s, dict) and s.get("audio_rendered") is False:
                        continue
                    n_stems += 1
                n_stems = n_stems or len(stems)
            dur = float(sf.info(str(mix)).duration)
            rows.append(
                {
                    "corpus": "slakh",
                    "split": "train" if split == "omitted" else split,
                    "song_id": track.name,
                    "mix_path": str(mix),
                    "midi_path": str(midi) if midi.is_file() else "",
                    "n_stems": int(n_stems),
                    "duration_sec": dur,
                }
            )
    return pd.DataFrame(rows)


def hour_matched_subsample(df: pd.DataFrame, *, target_hours: float, seed: int) -> pd.DataFrame:
    """Greedy subsample train split to ~target_hours of song duration."""
    train = df[df["split"] == "train"].copy()
    other = df[df["split"] != "train"]
    train = train.sample(frac=1.0, random_state=seed)
    kept = []
    hours = 0.0
    for _, row in train.iterrows():
        dur = row.get("duration_sec")
        if dur is None or (isinstance(dur, float) and pd.isna(dur)):
            continue
        kept.append(row)
        hours += float(dur) / 3600.0
        if hours >= target_hours:
            break
    sub = pd.DataFrame(kept)
    return pd.concat([sub, other], ignore_index=True)


def write_arm_manifests(df: pd.DataFrame, out_dir: Path, arm: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"manifest_{arm}.csv"
    df.to_csv(path, index=False)
    summary = {
        "arm": arm,
        "n_rows": int(len(df)),
        "by_split": df.groupby("split").size().to_dict() if len(df) else {},
        "hours": float(pd.to_numeric(df["duration_sec"], errors="coerce").sum() / 3600.0)
        if "duration_sec" in df.columns
        else None,
    }
    (out_dir / f"manifest_{arm}.summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return path


def _resolve_manifest_dir(cfg: dict) -> Path:
    try:
        out = resolve_dev_dir(cfg) / "manifests"
        out.mkdir(parents=True, exist_ok=True)
        return out
    except OSError:
        out = REPO_ROOT / "analysis" / "paper_data" / "transcription_manifests"
        out.mkdir(parents=True, exist_ok=True)
        return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--spdmx-root", type=Path, default=None)
    parser.add_argument("--slakh-root", type=Path, default=None)
    parser.add_argument("--hour-matched", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    out = _resolve_manifest_dir(cfg)
    spdmx_root = Path(args.spdmx_root or cfg.get("spdmx_root") or SPDMX_ROOT)
    slakh_root = Path(args.slakh_root or cfg.get("slakh_root") or SLAKH_ROOT)
    min_stems = int(cfg.get("min_stems", 2))
    seed = int(cfg.get("seed", 43))

    print(f"indexing SPDMX multitrack under {spdmx_root} …")
    spdmx = index_spdmx(
        spdmx_root,
        min_stems=min_stems,
        seed=seed,
        val_frac=float(cfg.get("val_fraction", 0.05)),
        test_frac=float(cfg.get("test_fraction", 0.05)),
    )
    write_arm_manifests(spdmx, out, "spdmx")
    print(f"  SPDMX rows={len(spdmx)}")

    if args.hour_matched or cfg.get("hour_matched_hours"):
        target = float(cfg.get("hour_matched_hours", 145.0))
        matched = hour_matched_subsample(spdmx, target_hours=target, seed=seed)
        write_arm_manifests(matched, out, "spdmx_hour_matched")
        print(f"  SPDMX hour-matched (~{target}h) rows={len(matched)}")

    print(f"indexing Slakh under {slakh_root} …")
    slakh = index_slakh(
        slakh_root,
        include_omitted=bool(cfg.get("slakh_include_omitted", False)),
    )
    write_arm_manifests(slakh, out, "slakh")
    print(f"  Slakh rows={len(slakh)}")

    both = pd.concat([slakh, spdmx], ignore_index=True)
    write_arm_manifests(both, out, "both")
    print(f"wrote manifests under {out}")


if __name__ == "__main__":
    main()
