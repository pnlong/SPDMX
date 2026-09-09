"""Freeze Slakh vs sPDMX (BDGP-eligible) manifests from pack_index.csv.

Packs are already filtered to songs with bass/drums/guitar/piano, so the
sPDMX arm is that full eligible pool (no hour-matched vs full split).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from experiments.separation.paths import load_config, resolve_dev_dir


def freeze_manifests(
    index: pd.DataFrame,
    out_dir: Path,
    *,
    seed: int,
    val_fraction: float,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    slakh_train = index[(index["corpus"] == "slakh") & (index["split"] == "train")].copy()
    slakh_val = index[(index["corpus"] == "slakh") & (index["split"] == "validation")].copy()
    slakh_test = index[(index["corpus"] == "slakh") & (index["split"] == "test")].copy()
    spdmx = index[index["corpus"] == "spdmx"].copy()
    if spdmx.empty:
        raise RuntimeError("no spdmx rows in pack_index")

    spdmx = spdmx.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_val = max(1, int(round(len(spdmx) * val_fraction)))
    spdmx_val = spdmx.iloc[:n_val].copy()
    spdmx_train = spdmx.iloc[n_val:].copy()

    manifests = {
        "slakh": {
            "train": slakh_train,
            "val": slakh_val,
            "test": slakh_test,
        },
        "spdmx": {
            "train": spdmx_train,
            "val": spdmx_val,
            "test": slakh_test,  # primary eval is Slakh2100 test for all arms
        },
    }

    summary: dict = {
        "seed": seed,
        "slakh_train_hours": float(slakh_train["hours"].sum()) if len(slakh_train) else 0.0,
        "spdmx_train_hours": float(spdmx_train["hours"].sum()) if len(spdmx_train) else 0.0,
        "note": "sPDMX arm = BDGP-eligible packs only (same filter as prepare_stems)",
        "arms": {},
    }

    for arm, splits in manifests.items():
        arm_dir = out_dir / arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        arm_info = {}
        for split_name, df in splits.items():
            path = arm_dir / f"{split_name}.csv"
            df.to_csv(path, index=False)
            arm_info[split_name] = {
                "n": int(len(df)),
                "hours": float(df["hours"].sum()) if len(df) else 0.0,
                "path": str(path),
            }
        summary["arms"][arm] = arm_info

    with open(out_dir / "manifest_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--pack-index", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    packs = resolve_dev_dir(cfg) / "packs"
    index_path = args.pack_index or (packs / "pack_index.csv")
    if not index_path.is_file():
        raise SystemExit(f"missing pack index: {index_path}")
    index = pd.read_csv(index_path)
    out_dir = args.out or (resolve_dev_dir(cfg) / "manifests")
    summary = freeze_manifests(
        index,
        out_dir,
        seed=int(cfg.get("seed", 43)),
        val_fraction=float(cfg.get("spdmx_val_fraction", 0.05)),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
