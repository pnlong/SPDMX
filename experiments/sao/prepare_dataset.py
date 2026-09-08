"""Build audio+caption JSONL datasets for SAO fine-tunes (three arms)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from experiments.sao.paths import (
    ARMS,
    SLAKH_ROOT,
    load_config,
    resolve_dev_dir,
)
from experiments.separation.paths import resolve_dev_dir as sep_dev_dir
from experiments.separation.targets import slakh_inst_class_to_target


def _caption(instruments: list[str], template: str) -> str:
    uniq = sorted(set(instruments)) or ["ensemble"]
    return template.format(instruments=", ".join(uniq))


def _slakh_mix_and_instruments(track_dir: Path) -> tuple[Path | None, list[str]]:
    mix = track_dir / "mix.flac"
    if not mix.is_file():
        return None, []
    meta_path = track_dir / "metadata.yaml"
    instruments: list[str] = []
    if meta_path.is_file():
        with open(meta_path) as f:
            meta = yaml.safe_load(f) or {}
        for sm in (meta.get("stems") or {}).values():
            t = slakh_inst_class_to_target(sm.get("inst_class"), is_drum=bool(sm.get("is_drum")))
            if t:
                instruments.append(t)
    return mix, instruments


def build_from_separation_manifests(
    *,
    cfg: dict,
    out_dir: Path,
) -> dict:
    sep_root = sep_dev_dir()
    packs = sep_root / "packs"
    manifests = sep_root / "manifests"
    template = str(cfg.get("caption_template") or "instrumental music, {instruments}")
    summary = {}

    for arm in ARMS:
        train_csv = manifests / arm / "train.csv"
        if not train_csv.is_file():
            print(f"skip {arm}: missing {train_csv}")
            continue
        df = pd.read_csv(train_csv)
        records = []
        for _, row in df.iterrows():
            mix_path = packs / row["path"] / "mix.flac"
            if not mix_path.is_file():
                continue
            # Instruments from target filenames present in pack.
            instruments = [
                t
                for t in ("bass", "drums", "guitar", "piano")
                if (packs / row["path"] / f"{t}.flac").is_file()
            ]
            records.append(
                {
                    "path": str(mix_path.resolve()),
                    "caption": _caption(instruments, template),
                    "song_id": row["song_id"],
                    "corpus": row.get("corpus", arm),
                }
            )
        arm_dir = out_dir / arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        jsonl = arm_dir / "train.jsonl"
        with open(jsonl, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        # stable-audio-tools style custom metadata JSON (path → prompt)
        meta = {rec["path"]: {"prompt": rec["caption"]} for rec in records}
        with open(arm_dir / "dataset_meta.json", "w") as f:
            json.dump(meta, f)
        custom_meta = Path(__file__).resolve().parent / "custom_metadata.py"
        # Dataset config for stable-audio-tools
        ds_cfg = {
            "dataset_type": "audio_dir",
            "datasets": [
                {
                    "id": arm,
                    "path": str(arm_dir / "audio_links"),
                    "custom_metadata_module": str(custom_meta),
                    "custom_metadata_args": {"meta_json": str(arm_dir / "dataset_meta.json")},
                }
            ],
            "random_crop": True,
        }
        # Symlink mixes into a flat audio_dir for audio_dir dataset type.
        link_root = arm_dir / "audio_links"
        link_root.mkdir(parents=True, exist_ok=True)
        for i, rec in enumerate(records):
            src = Path(rec["path"])
            dst = link_root / f"{i:06d}_{src.name}"
            if not dst.exists():
                try:
                    dst.symlink_to(src)
                except OSError:
                    pass
        with open(arm_dir / "dataset_config.json", "w") as f:
            json.dump(ds_cfg, f, indent=2)
        summary[arm] = {"n": len(records), "jsonl": str(jsonl)}
    return summary


def build_slakh_fallback(cfg: dict, out_dir: Path) -> dict:
    """If separation packs are missing, index Slakh mixes directly."""
    template = str(cfg.get("caption_template") or "instrumental music, {instruments}")
    records = []
    for split in ("train",):
        split_dir = SLAKH_ROOT / split
        if not split_dir.is_dir():
            continue
        for track_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            mix, instruments = _slakh_mix_and_instruments(track_dir)
            if mix is None:
                continue
            records.append(
                {
                    "path": str(mix.resolve()),
                    "caption": _caption(instruments, template),
                    "song_id": track_dir.name,
                    "corpus": "slakh",
                }
            )
    arm_dir = out_dir / "slakh"
    arm_dir.mkdir(parents=True, exist_ok=True)
    with open(arm_dir / "train.jsonl", "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    return {"slakh": {"n": len(records)}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    out_dir = args.out or (resolve_dev_dir(cfg) / "datasets")
    out_dir.mkdir(parents=True, exist_ok=True)

    if cfg.get("reuse_separation_manifests", True):
        summary = build_from_separation_manifests(cfg=cfg, out_dir=out_dir)
        if not summary:
            summary = build_slakh_fallback(cfg, out_dir)
    else:
        summary = build_slakh_fallback(cfg, out_dir)

    with open(out_dir / "prepare_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
