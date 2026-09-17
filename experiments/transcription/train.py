"""Train / eval entrypoints for transcription C1 (YourMT3 wiring).

This module does not vendor YourMT3. It documents the matched-step arms and can
emit a paper-ready CSV template. Wire YourMT3 training against the manifests
from ``prepare_manifest``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from experiments.transcription.paths import ARMS, load_config, resolve_dev_dir

REPO_PAPER = Path(__file__).resolve().parents[2] / "analysis" / "paper_data"


def write_placeholder_metrics(out_csv: Path) -> None:
    """Stub metrics table until YourMT3 runs finish."""
    rows = [
        {
            "arm": "slakh",
            "test_set": "slakh",
            "note_f1_onset": None,
            "note_f1_onset_offset": None,
            "status": "pending",
        },
        {
            "arm": "spdmx",
            "test_set": "slakh",
            "note_f1_onset": None,
            "note_f1_onset_offset": None,
            "status": "pending",
        },
        {
            "arm": "both",
            "test_set": "slakh",
            "note_f1_onset": None,
            "note_f1_onset_offset": None,
            "status": "pending",
        },
        {
            "arm": "spdmx",
            "test_set": "spdmx",
            "note_f1_onset": None,
            "note_f1_onset_offset": None,
            "status": "pending",
        },
    ]
    df = pd.DataFrame(rows)
    paper = REPO_PAPER / "transcription_note_f1.csv"
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
    parser.add_argument(
        "--arm",
        choices=(*ARMS, "all"),
        default="all",
        help="Which manifest arm to train (requires YourMT3 install)",
    )
    parser.add_argument(
        "--write-placeholder-metrics",
        action="store_true",
        help="Write pending metrics CSV for paper/blog scaffolding",
    )
    parser.add_argument(
        "--yourmt3-root",
        type=Path,
        default=None,
        help="Path to cloned https://github.com/mimbres/YourMT3 (optional)",
    )
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    dev = resolve_dev_dir(cfg)
    manifests = dev / "manifests"

    if args.write_placeholder_metrics:
        write_placeholder_metrics(dev / "metrics" / "transcription_note_f1.csv")
        return

    arms = list(ARMS) if args.arm == "all" else [args.arm]
    for arm in arms:
        path = manifests / f"manifest_{arm}.csv"
        if not path.is_file():
            raise SystemExit(
                f"missing {path}; run: python -m experiments.transcription.prepare_manifest"
            )
        print(f"arm={arm} manifest={path} rows={sum(1 for _ in open(path)) - 1}")

    if args.yourmt3_root is None or not args.yourmt3_root.is_dir():
        print(
            "YourMT3 not configured. Clone https://github.com/mimbres/YourMT3 and pass "
            "--yourmt3-root, or train externally using the CSV manifests "
            "(mix_path, midi_path, split, n_stems)."
        )
        print(f"Matched steps from config: {cfg.get('train_steps', 100000)}")
        write_placeholder_metrics(dev / "metrics" / "transcription_note_f1.csv")
        return

    # Hook for a future YourMT3 trainer launch.
    meta = {
        "yourmt3_root": str(args.yourmt3_root),
        "arms": arms,
        "train_steps": cfg.get("train_steps", 100000),
        "manifests": str(manifests),
    }
    (dev / "train_launch.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {dev / 'train_launch.json'} — wire YourMT3 CLI to these manifests next")


if __name__ == "__main__":
    main()
