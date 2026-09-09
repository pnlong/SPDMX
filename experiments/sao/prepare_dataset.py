"""Build mix+caption datasets for SAO fine-tunes.

Arms:
  - slakh: Slakh train mixes
  - spdmx_matched: same BDGP-eligible sPDMX song pool as separation
  - spdmx_full: all sPDMX songs with on-disk stems (corpus view)
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import yaml
from tqdm import tqdm

from experiments.sao.paths import (
    ARMS,
    SLAKH_ROOT,
    SPDMX_ROOT,
    load_config,
    resolve_dev_dir,
)
from experiments.separation.audio_io import audio_duration_seconds, load_mono, sum_stems
from experiments.separation.paths import TARGETS, resolve_dev_dir as sep_dev_dir
from experiments.separation.targets import gm_to_target, slakh_inst_class_to_target
from synthesis.patches import patch_group_key


def _caption(instruments: list[str], template: str) -> str:
    uniq = sorted(set(instruments)) or ["ensemble"]
    return template.format(instruments=", ".join(uniq))


def _write_stereo_flac(path: Path, mono: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mono = np.asarray(mono, dtype=np.float32)
    stereo = np.stack([mono, mono], axis=1)  # (T, 2)
    sf.write(str(path), stereo, sample_rate, format="FLAC")


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
                "stem_paths": "",
                "caption": _caption(instruments, template),
                "duration_sec": dur,
                "hours": dur / 3600.0,
            }
        )
    return pd.DataFrame(rows)


def _spdmx_song_dir(spdmx_root: Path, row: pd.Series) -> Path | None:
    path_col = row.get("path")
    if path_col is None or (isinstance(path_col, float) and pd.isna(path_col)):
        return None
    rel = str(path_col).replace("\\", "/").lstrip("./")
    return spdmx_root / rel


def index_spdmx(
    *,
    spdmx_root: Path,
    template: str,
    csv_name: str = "stems.csv",
    max_songs: int | None = None,
) -> pd.DataFrame:
    """Index every sPDMX song with on-disk stems; flag BDGP-eligible rows."""
    csv_path = spdmx_root / csv_name
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
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
        song_dir = _spdmx_song_dir(spdmx_root, g.iloc[0])
        if song_dir is None or not song_dir.is_dir():
            continue
        stem_paths: list[Path] = []
        bdgp_present: set[str] = set()
        for _, r in g.iterrows():
            cand = song_dir / f"{int(r['track'])}.flac"
            if not cand.is_file():
                continue
            stem_paths.append(cand)
            t = gm_to_target(int(r["program"]), bool(r["is_drum"]))
            if t is not None:
                bdgp_present.add(t)
        if not stem_paths:
            stem_paths = sorted(song_dir.glob("*.flac"))
        if not stem_paths:
            continue
        try:
            dur = audio_duration_seconds(stem_paths[0])
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
        rows.append(
            {
                "corpus": "spdmx",
                "song_id": str(song_id),
                "mix_path": "",
                "stem_paths": json.dumps([str(p) for p in stem_paths]),
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


def _render_one_mix(payload: dict) -> dict | None:
    """Worker: sum sPDMX stems → stereo mix.flac."""
    dest = Path(payload["dest"])
    stems = [Path(p) for p in payload["stems"]]
    sample_rate = int(payload["sample_rate"])
    song_id = payload["song_id"]
    if dest.is_file():
        return {
            "song_id": song_id,
            "mix_path": str(dest.resolve()),
            "duration_sec": audio_duration_seconds(dest),
        }
    try:
        arrays = []
        for p in stems:
            a, _ = load_mono(p, sample_rate=sample_rate)
            arrays.append(a)
        mix = sum_stems(arrays)
        _write_stereo_flac(dest, mix, sample_rate)
        return {
            "song_id": song_id,
            "mix_path": str(dest.resolve()),
            "duration_sec": float(mix.shape[0]) / float(sample_rate),
        }
    except Exception as exc:  # noqa: BLE001
        print(f"skip mix {song_id}: {exc}")
        return None


def render_spdmx_mixes(
    df: pd.DataFrame,
    *,
    mixes_root: Path,
    sample_rate: int,
    jobs: int,
) -> pd.DataFrame:
    """Render missing mixes for the given sPDMX rows; return df with mix_path set."""
    if df.empty:
        return df
    jobs_payload: list[dict] = []
    for _, row in df.iterrows():
        song_id = str(row["song_id"])
        dest = mixes_root / song_id / "mix.flac"
        stems = json.loads(row["stem_paths"]) if row["stem_paths"] else []
        if not stems:
            continue
        jobs_payload.append(
            {
                "dest": str(dest),
                "stems": stems,
                "sample_rate": sample_rate,
                "song_id": song_id,
            }
        )

    results: dict[str, dict] = {}
    if jobs <= 1:
        for payload in tqdm(jobs_payload, desc="sao:render-mixes"):
            row = _render_one_mix(payload)
            if row:
                results[row["song_id"]] = row
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futs = [pool.submit(_render_one_mix, p) for p in jobs_payload]
            for fut in tqdm(as_completed(futs), total=len(futs), desc="sao:render-mixes"):
                row = fut.result()
                if row:
                    results[row["song_id"]] = row

    out_rows = []
    for _, row in df.iterrows():
        sid = str(row["song_id"])
        if sid not in results:
            continue
        rec = row.to_dict()
        rec["mix_path"] = results[sid]["mix_path"]
        rec["duration_sec"] = float(results[sid]["duration_sec"])
        rec["hours"] = rec["duration_sec"] / 3600.0
        out_rows.append(rec)
    return pd.DataFrame(out_rows)


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

    ds_cfg = {
        "dataset_type": "audio_dir",
        "datasets": [
            {
                "id": arm,
                "path": str(link_root),
                "custom_metadata_module": str(custom_meta),
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
    sample_rate = int(cfg.get("sample_rate", 44100))
    jobs = max(1, int(cfg.get("render_jobs") or min(8, (os.cpu_count() or 4))))
    slakh_root = Path(cfg.get("slakh_root") or SLAKH_ROOT)
    spdmx_root = Path(cfg.get("spdmx_root") or SPDMX_ROOT)
    max_spdmx = cfg.get("spdmx_index_max_songs")
    max_spdmx_i = int(max_spdmx) if max_spdmx is not None else None

    custom_meta = Path(__file__).resolve().parent / "custom_metadata.py"
    mixes_root = out_dir / "mixes" / "spdmx"
    mixes_root.mkdir(parents=True, exist_ok=True)

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
        raise RuntimeError(f"no sPDMX songs found under {spdmx_root}")

    matched = _bdgp_matched_pool(spdmx, seed=seed, spdmx_root=spdmx_root)
    if matched.empty:
        raise RuntimeError(
            "no BDGP-eligible sPDMX songs for matched arm; "
            "run separation prepare_stems/freeze_manifests or check indexing"
        )
    matched_ids = set(matched["song_id"].astype(str))

    print(f"sao: rendering BDGP-matched mixes ({len(matched)} songs, jobs={jobs})")
    matched_rendered = render_spdmx_mixes(
        matched,
        mixes_root=mixes_root,
        sample_rate=sample_rate,
        jobs=jobs,
    )
    if matched_rendered.empty:
        raise RuntimeError("failed to render any matched sPDMX mixes")

    remaining = spdmx[~spdmx["song_id"].astype(str).isin(matched_ids)].copy()
    print(f"sao: rendering remaining full-pool mixes ({len(remaining)} songs)")
    rest_rendered = render_spdmx_mixes(
        remaining,
        mixes_root=mixes_root,
        sample_rate=sample_rate,
        jobs=jobs,
    )
    full_rendered = pd.concat([matched_rendered, rest_rendered], ignore_index=True)

    arms = {
        "slakh": slakh,
        "spdmx_matched": matched_rendered,
        "spdmx_full": full_rendered,
    }
    summary: dict = {
        "slakh_train_hours": slakh_hours,
        "spdmx_indexed": int(len(spdmx)),
        "spdmx_bdgp_eligible": int(spdmx["bdgp_eligible"].sum()) if "bdgp_eligible" in spdmx else None,
        "spdmx_matched_hours": float(matched_rendered["hours"].sum()),
        "spdmx_full_hours": float(full_rendered["hours"].sum()),
        "spdmx_rendered": int(len(full_rendered)),
        "note": "spdmx_matched = BDGP-eligible (sep-aligned); spdmx_full = all songs",
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
    parser.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=None,
        help="Parallel mix-render workers (overrides config render_jobs)",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.jobs is not None:
        cfg = {**cfg, "render_jobs": args.jobs}
    out_dir = args.out or (resolve_dev_dir(cfg) / "datasets")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = prepare_all(cfg, out_dir)
    with open(out_dir / "prepare_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
