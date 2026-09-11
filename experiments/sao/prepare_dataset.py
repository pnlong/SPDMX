"""Build mix+caption datasets for SAO fine-tunes.

Arms:
  - slakh: Slakh train mixes
  - spdmx_matched: same BDGP-eligible sPDMX song pool as separation
  - spdmx_full: all sPDMX songs with a shipped mix

sPDMX arms use dataset mixes (``mix.flac`` / ``SPDMX_dev/mix/``) directly —
they do **not** re-sum stems. Mixes are **mono**; SAO training duplicates to
stereo (L=R) via stable-audio-tools ``Stereo()`` when ``audio_channels: 2``.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

from experiments.sao.paths import (
    ARMS,
    SLAKH_ROOT,
    SPDMX_ROOT,
    load_config,
    resolve_dev_dir,
)
from experiments.separation.audio_io import audio_duration_seconds
from experiments.separation.paths import TARGETS, resolve_dev_dir as sep_dev_dir
from experiments.separation.targets import gm_to_target, slakh_inst_class_to_target
from synthesis.patches import patch_group_key


def _caption(instruments: list[str], template: str) -> str:
    uniq = sorted(set(instruments)) or ["ensemble"]
    return template.format(instruments=", ".join(uniq))


def _slakh_mix_and_instruments(track_dir: Path) -> tuple[Path | None, list[str], float]:
    mix = track_dir / "mix.flac"
    if not mix.is_file():
        return None, [], 0.0
    instruments: list[str] = []
    meta_path = track_dir / "metadata.yaml"
    if meta_path.is_file():
        with open(meta_path) as f:
            meta = yaml.safe_load(f) or {}
        for sm in (meta.get("stems") or {}).values():
            t = slakh_inst_class_to_target(sm.get("inst_class"), is_drum=bool(sm.get("is_drum")))
            if t:
                instruments.append(t)
    dur = audio_duration_seconds(mix)
    return mix, instruments, dur


def index_slakh_train(*, slakh_root: Path, template: str) -> pd.DataFrame:
    """Index native Slakh train mixes (no remapping)."""
    rows: list[dict] = []
    split_dir = slakh_root / "train"
    if not split_dir.is_dir():
        raise FileNotFoundError(f"missing Slakh train dir: {split_dir}")
    track_dirs = sorted(p for p in split_dir.iterdir() if p.is_dir())
    for track_dir in tqdm(track_dirs, desc="sao:index-slakh"):
        mix, instruments, dur = _slakh_mix_and_instruments(track_dir)
        if mix is None or dur <= 0:
            continue
        rows.append(
            {
                "corpus": "slakh",
                "song_id": track_dir.name,
                "mix_path": str(mix.resolve()),
                "caption": _caption(instruments, template),
                "duration_sec": dur,
                "hours": dur / 3600.0,
            }
        )
    return pd.DataFrame(rows)


def _mix_path_from_songs_row(spdmx_root: Path, row: pd.Series) -> Path | None:
    """Resolve mix path from a songs.csv row."""
    from experiments.separation.spdmx_io import resolve_spdmx_mix

    return resolve_spdmx_mix(spdmx_root, row, song_id=str(row["song_id"]))


def _index_spdmx_from_songs(
    *,
    spdmx_root: Path,
    template: str,
    max_songs: int | None = None,
) -> pd.DataFrame | None:
    """Fast path: read ``songs.csv`` (needs ``song_length`` + ``mix`` / ``path``)."""
    from synthesis.build_songs_table import SONG_LENGTH_COLUMN, SUBSET_BDGP

    songs_path = spdmx_root / "songs.csv"
    if not songs_path.is_file():
        return None
    songs = pd.read_csv(songs_path)
    if SONG_LENGTH_COLUMN not in songs.columns:
        return None
    if "chunk" in songs.columns:
        songs["chunk"] = songs["chunk"].map(lambda x: x if pd.isna(x) else str(int(x)))

    rows: list[dict] = []
    for _, row in tqdm(songs.iterrows(), total=len(songs), desc="sao:index-spdmx"):
        if max_songs is not None and len(rows) >= max_songs:
            break
        dur = pd.to_numeric(row.get(SONG_LENGTH_COLUMN), errors="coerce")
        if pd.isna(dur) or float(dur) <= 0:
            continue
        mix = _mix_path_from_songs_row(spdmx_root, row)
        if mix is None:
            continue
        gm_raw = row.get("gm_classes")
        instruments = (
            [p for p in str(gm_raw).split("|") if p and p.lower() != "nan"]
            if pd.notna(gm_raw)
            else []
        )
        bdgp = row.get(SUBSET_BDGP, False)
        if isinstance(bdgp, str):
            bdgp = bdgp.lower() in ("true", "1", "yes")
        rows.append(
            {
                "corpus": "spdmx",
                "song_id": str(row["song_id"]),
                "mix_path": str(mix.resolve()),
                "caption": _caption(instruments, template),
                "duration_sec": float(dur),
                "hours": float(dur) / 3600.0,
                "bdgp_eligible": bool(bdgp),
            }
        )
    print(f"sao: indexed {len(rows)} songs from songs.csv (song_length)")
    return pd.DataFrame(rows)


def index_spdmx(
    *,
    spdmx_root: Path,
    template: str,
    csv_name: str = "stems.csv",
    max_songs: int | None = None,
) -> pd.DataFrame:
    """Index sPDMX songs that already have a shipped mix.

    Prefers ``songs.csv`` when ``song_length`` is present (no FLAC opens). Falls
    back to scanning ``stems.csv`` + mix headers otherwise.
    """
    from experiments.separation.spdmx_io import resolve_spdmx_mix

    fast = _index_spdmx_from_songs(
        spdmx_root=spdmx_root, template=template, max_songs=max_songs,
    )
    if fast is not None and not fast.empty:
        return fast

    csv_path = spdmx_root / csv_name
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    print("sao: songs.csv missing song_length; falling back to stems.csv + mix headers")
    df = pd.read_csv(csv_path)
    if "chunk" in df.columns:
        df["chunk"] = df["chunk"].map(lambda x: x if pd.isna(x) else str(int(x)))
    if "is_drum" in df.columns:
        df["is_drum"] = df["is_drum"].astype(str).str.lower().isin(("true", "1", "yes"))

    rows: list[dict] = []
    grouped = df.groupby("song_id", sort=False)
    n_songs = int(df["song_id"].nunique())
    for song_id, g in tqdm(grouped, desc="sao:index-spdmx", total=n_songs):
        if max_songs is not None and len(rows) >= max_songs:
            break
        song_id_s = str(song_id)
        head = g.iloc[0]
        shipped = resolve_spdmx_mix(spdmx_root, head, song_id=song_id_s)
        if shipped is None:
            continue
        try:
            dur = audio_duration_seconds(shipped)
        except Exception:  # noqa: BLE001
            continue
        if dur <= 0:
            continue

        instruments = sorted(
            {
                patch_group_key(int(r["program"]), bool(r["is_drum"]))
                for _, r in g.iterrows()
            }
        )
        bdgp_present: set[str] = set()
        for _, r in g.iterrows():
            t = gm_to_target(int(r["program"]), bool(r["is_drum"]))
            if t is not None:
                bdgp_present.add(t)
        rows.append(
            {
                "corpus": "spdmx",
                "song_id": song_id_s,
                "mix_path": str(shipped.resolve()),
                "caption": _caption(instruments, template),
                "duration_sec": dur,
                "hours": dur / 3600.0,
                "bdgp_eligible": set(TARGETS).issubset(bdgp_present),
            }
        )
    return pd.DataFrame(rows)


def _bdgp_matched_pool(spdmx: pd.DataFrame, *, seed: int, spdmx_root: Path) -> pd.DataFrame:
    """BDGP-eligible songs via songs.csv ``subset:bdgp``, else sep manifest / recompute."""
    from synthesis.build_songs_table import load_bdgp_song_ids

    ids = load_bdgp_song_ids(spdmx_root)
    if ids:
        matched = spdmx[spdmx["song_id"].astype(str).isin(ids)].copy()
        if not matched.empty:
            print(f"sao: matched arm from songs.csv subset:bdgp (n={len(matched)})")
            return matched.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    sep_train = sep_dev_dir() / "manifests" / "spdmx" / "train.csv"
    if not sep_train.is_file():
        for legacy in ("spdmx_matched", "spdmx_full"):
            cand = sep_dev_dir() / "manifests" / legacy / "train.csv"
            if cand.is_file():
                sep_train = cand
                break
    if sep_train.is_file():
        sep_ids = set(pd.read_csv(sep_train)["song_id"].astype(str))
        matched = spdmx[spdmx["song_id"].astype(str).isin(sep_ids)].copy()
        if not matched.empty:
            print(f"sao: matched arm from separation manifest {sep_train} (n={len(matched)})")
            return matched.reset_index(drop=True)

    matched = spdmx[spdmx["bdgp_eligible"] == True].copy()  # noqa: E712
    print(f"sao: matched arm = BDGP-eligible index rows (n={len(matched)})")
    if matched.empty:
        return matched
    return matched.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _write_arm(
    arm: str,
    df: pd.DataFrame,
    *,
    out_dir: Path,
    custom_meta: Path,
) -> dict:
    arm_dir = out_dir / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for _, row in df.iterrows():
        mix = Path(row["mix_path"])
        if not mix.is_file():
            continue
        records.append(
            {
                "path": str(mix.resolve()),
                "caption": row["caption"],
                "song_id": row["song_id"],
                "corpus": row.get("corpus", arm),
                "duration_sec": float(row["duration_sec"]),
                "hours": float(row["hours"]),
            }
        )

    jsonl = arm_dir / "train.jsonl"
    with open(jsonl, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    meta = {rec["path"]: {"prompt": rec["caption"]} for rec in records}
    with open(arm_dir / "dataset_meta.json", "w") as f:
        json.dump(meta, f)

    link_root = arm_dir / "audio_links"
    link_root.mkdir(parents=True, exist_ok=True)
    # Drop stale links from a previous prepare.
    for old in link_root.iterdir():
        if old.is_symlink() or old.is_file():
            old.unlink()
    for i, rec in enumerate(records):
        src = Path(rec["path"])
        dst = link_root / f"{i:06d}_{src.parent.name}_{src.name}"
        try:
            dst.symlink_to(src)
        except OSError:
            pass

    # Copy hook into the shared arm dir so dataset_config does not depend on a checkout path.
    shared_meta = arm_dir / "custom_metadata.py"
    shutil.copy2(custom_meta, shared_meta)

    ds_cfg = {
        "dataset_type": "audio_dir",
        "datasets": [
            {
                "id": arm,
                "path": str(link_root),
                "custom_metadata_module": str(shared_meta.resolve()),
                "custom_metadata_args": {"meta_json": str(arm_dir / "dataset_meta.json")},
            }
        ],
        "random_crop": True,
    }
    with open(arm_dir / "dataset_config.json", "w") as f:
        json.dump(ds_cfg, f, indent=2)

    manifest_dir = out_dir / "manifests" / arm
    manifest_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(manifest_dir / "train.csv", index=False)

    return {
        "n": len(records),
        "hours": float(sum(r["hours"] for r in records)),
        "jsonl": str(jsonl),
        "dataset_config": str(arm_dir / "dataset_config.json"),
    }


def prepare_all(cfg: dict, out_dir: Path) -> dict:
    template = str(cfg.get("caption_template") or "instrumental music, {instruments}")
    seed = int(cfg.get("seed", 43))
    slakh_root = Path(cfg.get("slakh_root") or SLAKH_ROOT)
    spdmx_root = Path(cfg.get("spdmx_root") or SPDMX_ROOT)
    max_spdmx = cfg.get("spdmx_index_max_songs")
    max_spdmx_i = int(max_spdmx) if max_spdmx is not None else None

    custom_meta = Path(__file__).resolve().parent / "custom_metadata.py"

    slakh = index_slakh_train(slakh_root=slakh_root, template=template)
    if slakh.empty:
        raise RuntimeError("no Slakh train mixes found")
    slakh_hours = float(slakh["hours"].sum())

    spdmx = index_spdmx(
        spdmx_root=spdmx_root,
        template=template,
        max_songs=max_spdmx_i,
    )
    if spdmx.empty:
        raise RuntimeError(
            f"no sPDMX songs with shipped mixes under {spdmx_root}; "
            "run synthesis.final --only-pass song_mix (or build_spdmx) first"
        )

    matched = _bdgp_matched_pool(spdmx, seed=seed, spdmx_root=spdmx_root)
    if matched.empty:
        raise RuntimeError(
            "no BDGP-eligible sPDMX songs with shipped mixes for matched arm; "
            "check songs.csv subset:bdgp and mix/ files"
        )
    matched_ids = set(matched["song_id"].astype(str))
    remaining = spdmx[~spdmx["song_id"].astype(str).isin(matched_ids)].copy()
    print(
        f"sao: using shipped mixes "
        f"(matched={len(matched)}, full={len(spdmx)}, remaining={len(remaining)})"
    )

    arms = {
        "slakh": slakh,
        "spdmx_matched": matched,
        "spdmx_full": spdmx,
    }
    summary: dict = {
        "slakh_train_hours": slakh_hours,
        "spdmx_indexed": int(len(spdmx)),
        "spdmx_bdgp_eligible": int(spdmx["bdgp_eligible"].sum()) if "bdgp_eligible" in spdmx else None,
        "spdmx_matched_hours": float(matched["hours"].sum()),
        "spdmx_full_hours": float(spdmx["hours"].sum()),
        "note": (
            "spdmx_matched = BDGP-eligible; spdmx_full = all songs with shipped "
            "mix.flac / SPDMX_dev/mix/ (no stem re-sum)"
        ),
        "arms": {},
    }
    for arm in ARMS:
        summary["arms"][arm] = _write_arm(
            arm, arms[arm], out_dir=out_dir, custom_meta=custom_meta
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    out_dir = args.out or (resolve_dev_dir(cfg) / "datasets")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = prepare_all(cfg, out_dir)
    with open(out_dir / "prepare_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
