"""Launch YourMT3 training for a transcription arm.

``slakh_run`` / ``spdmx_run`` in upstream docs are only YourMT3's required
``exp_id`` (checkpoint / W&B run name). Prefer this wrapper:

    uv run python -m experiments.transcription.train --arm slakh --gpu 1
    uv run python -m experiments.transcription.train --arm spdmx --gpu 2
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from experiments.transcription.paths import ARMS, TRANS_DIR, load_config, resolve_dev_dir

REPO_ROOT = Path(__file__).resolve().parents[2]
REPO_PAPER = REPO_ROOT / "analysis" / "paper_data"
YOURMT3_SRC = TRANS_DIR / "YourMT3" / "amt" / "src"

# Manifest arm → YourMT3 data preset (see manifest_to_yourmt3_indexes.ARM_TO_DATASET).
ARM_TO_PRESET = {
    "slakh": "slakh_redux",
    "spdmx": "spdmx",
    "both": "slakh_spdmx",
}


def write_placeholder_metrics(out_csv: Path) -> None:
    """Stub metrics table until YourMT3 runs finish."""
    rows = [
        {"arm": "slakh", "test_set": "slakh", "note_f1_onset": None, "note_f1_onset_offset": None, "status": "pending"},
        {"arm": "spdmx", "test_set": "slakh", "note_f1_onset": None, "note_f1_onset_offset": None, "status": "pending"},
        {"arm": "both", "test_set": "slakh", "note_f1_onset": None, "note_f1_onset_offset": None, "status": "pending"},
        {"arm": "spdmx", "test_set": "spdmx", "note_f1_onset": None, "note_f1_onset_offset": None, "status": "pending"},
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


def _build_cmd(
    *,
    arm: str,
    exp_id: str,
    project: str,
    max_steps: int,
    val_interval: int,
    samples_per_epoch: int,
    limit_val_batches: int,
    wandb_mode: str,
    extra: list[str],
) -> list[str]:
    preset = ARM_TO_PRESET[arm]
    return [
        sys.executable,
        str(YOURMT3_SRC / "train.py"),
        exp_id,
        "-d",
        preset,
        "-p",
        project,
        "--max-steps",
        str(max_steps),
        "--val-interval",
        str(val_interval),
        "--train-num-samples-per-epoch",
        str(samples_per_epoch),
        "--limit-val-batches",
        str(limit_val_batches),
        "-wb",
        wandb_mode,
        *extra,
    ]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Train a transcription arm via YourMT3 (thin wrapper around amt/src/train.py)."
    )
    parser.add_argument(
        "--arm",
        choices=tuple(ARM_TO_PRESET),
        required=False,
        help="Which data arm to train",
    )
    parser.add_argument("--gpu", type=str, default=None, help="CUDA device id(s), sets CUDA_VISIBLE_DEVICES")
    parser.add_argument("--exp-id", type=str, default=None, help="YourMT3 run / checkpoint id (default: <arm>; reuse to auto-resume)")
    parser.add_argument("--project", type=str, default="transcription", help="W&B project name (-p)")
    parser.add_argument("--max-steps", type=int, default=None, help="Override config train_steps")
    parser.add_argument(
        "--val-interval",
        type=int,
        default=None,
        help="Validate every N steps (YourMT3 -vit; default: config val_interval_steps)",
    )
    parser.add_argument(
        "--samples-per-epoch",
        type=int,
        default=None,
        help="Train crops per epoch for sampler (YourMT3 -se; default: config samples_per_epoch)",
    )
    parser.add_argument(
        "--limit-val-batches",
        type=int,
        default=None,
        help="Max validation batches per val pass (default: config limit_val_batches)",
    )
    parser.add_argument(
        "--wandb",
        choices=("offline", "online", "disabled"),
        default="offline",
        help="W&B mode (-wb)",
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--write-placeholder-metrics",
        action="store_true",
        help="Only write pending metrics CSV for paper/blog scaffolding",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the YourMT3 command without running it",
    )
    parser.add_argument(
        "yourmt3_args",
        nargs=argparse.REMAINDER,
        help="Extra args passed through to YourMT3 train.py (prefix with --)",
    )
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    dev = resolve_dev_dir(cfg)

    if args.write_placeholder_metrics:
        write_placeholder_metrics(dev / "metrics" / "transcription_note_f1.csv")
        return

    if args.arm is None:
        parser.error("--arm is required (unless --write-placeholder-metrics)")

    if not (YOURMT3_SRC / "train.py").is_file():
        raise SystemExit(
            f"YourMT3 missing at {YOURMT3_SRC}. "
            "Run: uv run python -m experiments.transcription.setup_yourmt3"
        )

    from experiments.transcription.patch_yourmt3 import apply_yourmt3_patches

    for p in apply_yourmt3_patches(YOURMT3_SRC):
        print(f"patched {p}")

    extra = list(args.yourmt3_args or [])
    if extra and extra[0] == "--":
        extra = extra[1:]

    max_steps = args.max_steps if args.max_steps is not None else int(cfg.get("train_steps", 100000))
    val_interval = (
        args.val_interval
        if args.val_interval is not None
        else int(cfg.get("val_interval_steps", 2000))
    )
    samples_per_epoch = (
        args.samples_per_epoch
        if args.samples_per_epoch is not None
        else int(cfg.get("samples_per_epoch", 90000))
    )
    limit_val_batches = (
        args.limit_val_batches
        if args.limit_val_batches is not None
        else int(cfg.get("limit_val_batches", 32))
    )
    exp_id = args.exp_id or args.arm
    cmd = _build_cmd(
        arm=args.arm,
        exp_id=exp_id,
        project=args.project,
        max_steps=max_steps,
        val_interval=val_interval,
        samples_per_epoch=samples_per_epoch,
        limit_val_batches=limit_val_batches,
        wandb_mode=args.wandb,
        extra=extra,
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{YOURMT3_SRC.resolve()}:{env.get('PYTHONPATH', '')}"
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.gpu

    print("cwd:", YOURMT3_SRC)
    print("cmd:", " ".join(cmd))
    if args.dry_run:
        return

    raise SystemExit(subprocess.call(cmd, cwd=str(YOURMT3_SRC), env=env))


if __name__ == "__main__":
    main()
