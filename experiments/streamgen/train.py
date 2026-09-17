"""StreamGen train/eval scaffolding (upstream stream-music-gen)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from experiments.streamgen.paths import ARMS, load_config, resolve_dev_dir

REPO_PAPER = Path(__file__).resolve().parents[2] / "analysis" / "paper_data"


def write_placeholder_metrics(out_csv: Path) -> None:
    rows = [
        {
            "arm": arm,
            "test_set": "slakh",
            "cocola": None,
            "beat_f1": None,
            "fad": None,
            "future_visibility": 0,
            "status": "pending",
        }
        for arm in ARMS
    ]
    rows.append(
        {
            "arm": "spdmx",
            "test_set": "spdmx",
            "cocola": None,
            "beat_f1": None,
            "fad": None,
            "future_visibility": 0,
            "status": "pending",
        }
    )
    df = pd.DataFrame(rows)
    paper = REPO_PAPER / "streamgen_metrics.csv"
    paper.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(paper, index=False)
    try:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        print(f"wrote placeholder {out_csv} and {paper}")
    except OSError as exc:
        print(f"wrote placeholder {paper} (dev dir skipped: {exc})")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--arm", choices=(*ARMS, "all"), default="all")
    parser.add_argument("--write-placeholder-metrics", action="store_true")
    parser.add_argument(
        "--upstream-root",
        type=Path,
        default=None,
        help="Path to cloned lukewys/stream-music-gen",
    )
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    dev = resolve_dev_dir(cfg)
    index = dev / "spdmx_index" / "spdmx_multitrack.jsonl"

    if args.write_placeholder_metrics:
        write_placeholder_metrics(dev / "metrics" / "streamgen_metrics.csv")
        return

    if not index.is_file():
        raise SystemExit(
            f"missing {index}; run: python -m experiments.streamgen.prepare_spdmx_index"
        )

    if args.upstream_root is None or not args.upstream_root.is_dir():
        print(
            "Clone https://github.com/lukewys/stream-music-gen and pass --upstream-root. "
            "See spdmx_index/ADAPTER.md for the SPDMX dataset adapter steps."
        )
        print(
            f"Pilot: future_visibility={cfg.get('future_visibility', 0)} "
            f"train_steps={cfg.get('train_steps', 100000)}"
        )
        write_placeholder_metrics(dev / "metrics" / "streamgen_metrics.csv")
        meta = {
            "index": str(index),
            "arms": list(ARMS) if args.arm == "all" else [args.arm],
            "config": cfg,
        }
        (dev / "train_launch.json").write_text(json.dumps(meta, indent=2) + "\n")
        return

    meta = {
        "upstream_root": str(args.upstream_root),
        "index": str(index),
        "arms": list(ARMS) if args.arm == "all" else [args.arm],
        "future_visibility": cfg.get("future_visibility", 0),
        "train_steps": cfg.get("train_steps", 100000),
    }
    (dev / "train_launch.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {dev / 'train_launch.json'}")


if __name__ == "__main__":
    main()
