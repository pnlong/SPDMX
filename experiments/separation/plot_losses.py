"""Plot train/val L1 curves from separation training losses.csv files."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from experiments.separation.paths import TRAIN_ARMS, load_config, resolve_dev_dir


def plot_losses(csv_path: Path, out_path: Path | None = None) -> Path:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise SystemExit(f"empty loss log: {csv_path}")

    fig, ax = plt.subplots(figsize=(8, 4))
    for split, color in (("train", "C0"), ("val", "C1")):
        sub = df[df["split"] == split]
        if sub.empty:
            continue
        ax.plot(sub["step"], sub["loss"], label=split, color=color, marker="o", markersize=2, linewidth=1.2)
    ax.set_xlabel("Step")
    ax.set_ylabel("L1 loss")
    ax.set_title(csv_path.parent.name)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = out_path or (csv_path.parent / "losses.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--arm",
        choices=(*TRAIN_ARMS, "all"),
        default="all",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Plot a specific losses.csv (overrides --arm)",
    )
    args = parser.parse_args()

    if args.csv is not None:
        print(plot_losses(args.csv))
        return

    root = resolve_dev_dir(load_config(args.config))
    arms = TRAIN_ARMS if args.arm == "all" else (args.arm,)
    for arm in arms:
        csv_path = root / "checkpoints" / arm / "losses.csv"
        if not csv_path.is_file():
            print(f"skip {arm}: missing {csv_path}")
            continue
        out = plot_losses(csv_path)
        print(f"{arm}: {out}")


if __name__ == "__main__":
    main()
