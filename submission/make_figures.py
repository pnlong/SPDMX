"""Regenerate ICASSP paper figures as transparent PDFs under submission/figs/."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from analysis.plots import (
    plot_ablation_listening,
    plot_chunk_layout,
    plot_downstream_poc,
    plot_gm_program_compare,
    plot_sao_metrics,
    plot_separation_sisdr,
)
from shared.config import OUTPUT_DIR
from synthesis.chunking import CHUNKS_FILE_NAME

FIGURES_DIR = Path(__file__).resolve().parent / "figs"
DATA_DIR = Path(__file__).resolve().parent / "data"
INSTRUMENTS_DIR = (
    Path(OUTPUT_DIR) / "dev" / "analysis" / "instruments" / "all_valid"
)
ORIGINAL_STEMS = INSTRUMENTS_DIR / "gm_program_stems.csv"
CORRECTED_STEMS = INSTRUMENTS_DIR / "gm_program_stems_corrected.csv"
CHUNKS_CSV = Path(OUTPUT_DIR) / "SPDMX" / CHUNKS_FILE_NAME


def make_gm_program_compare_figure(
    *,
    top_n: int = 10,
    rank_by: str = "corrected",
    show_percentages: bool = False,
) -> Path:
    original = pd.read_csv(ORIGINAL_STEMS)
    corrected = pd.read_csv(CORRECTED_STEMS)
    out = FIGURES_DIR / "gm_program_counts_compare.pdf"
    plot_gm_program_compare(
        original,
        corrected,
        out,
        top_n=top_n,
        rank_by=rank_by,
        show_percentages=show_percentages,
        figsize=(3.4, 3.2),
    )
    return out


def make_ablation_listening_figure() -> Path | None:
    csv_path = DATA_DIR / "ablation_listening.csv"
    if not csv_path.is_file():
        print(f"skip ablation figure: missing {csv_path}")
        return None
    out = FIGURES_DIR / "ablation_listening.pdf"
    plot_ablation_listening(pd.read_csv(csv_path), out, figsize=(5.5, 3.2))
    return out


def make_separation_figure() -> Path | None:
    csv_path = DATA_DIR / "separation_sisdr.csv"
    out = FIGURES_DIR / "separation_sisdr.pdf"
    if not csv_path.is_file():
        _placeholder_figure(out, "SI-SDR results pending")
        return out
    df = pd.read_csv(csv_path)
    if df.empty or df["si_sdr_mean"].isna().all():
        _placeholder_figure(out, "SI-SDR results pending")
        return out
    plot_separation_sisdr(df, out)
    return out


def make_sao_figure() -> Path | None:
    csv_path = DATA_DIR / "sao_metrics.csv"
    out = FIGURES_DIR / "sao_metrics.pdf"
    if not csv_path.is_file():
        _placeholder_figure(out, "FAD / CLAP results pending")
        return out
    df = pd.read_csv(csv_path)
    if df.empty or (df[["fad", "clap"]].isna().all(axis=None)):
        _placeholder_figure(out, "FAD / CLAP results pending")
        return out
    plot_sao_metrics(df, out)
    return out


def make_downstream_figure() -> Path:
    """Combined Demucs + SAO figure used in the camera-ready draft."""
    sep_path = DATA_DIR / "separation_sisdr.csv"
    sao_path = DATA_DIR / "sao_metrics.csv"
    out = FIGURES_DIR / "downstream_poc.pdf"
    sep = (
        pd.read_csv(sep_path)
        if sep_path.is_file()
        else pd.DataFrame(columns=["train_arm", "test_set", "target", "si_sdr_mean"])
    )
    sao = (
        pd.read_csv(sao_path)
        if sao_path.is_file()
        else pd.DataFrame(columns=["train_arm", "fad", "clap", "n"])
    )
    plot_downstream_poc(sep, sao, out)
    return out


def make_chunk_layout_figure() -> Path | None:
    if not CHUNKS_CSV.is_file():
        print(f"skip chunk layout figure: missing {CHUNKS_CSV}")
        return None
    out = FIGURES_DIR / "chunk_layout.pdf"
    plot_chunk_layout(pd.read_csv(CHUNKS_CSV), out)
    return out


def _placeholder_figure(path: Path, message: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.0, 2.5))
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=12)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", transparent=True)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument(
        "--rank-by",
        choices=("corrected", "original"),
        default="corrected",
    )
    parser.add_argument("--show-percentages", action="store_true")
    parser.add_argument(
        "--only",
        choices=("gm", "ablation", "separation", "sao", "downstream", "chunk", "all"),
        default="all",
    )
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    if args.only in ("gm", "all") and ORIGINAL_STEMS.is_file() and CORRECTED_STEMS.is_file():
        written.append(
            make_gm_program_compare_figure(
                top_n=args.top_n,
                rank_by=args.rank_by,
                show_percentages=args.show_percentages,
            )
        )
    elif args.only in ("gm", "all"):
        print(f"skip GM figure: missing {ORIGINAL_STEMS} or {CORRECTED_STEMS}")

    if args.only in ("ablation", "all"):
        p = make_ablation_listening_figure()
        if p:
            written.append(p)

    if args.only in ("separation", "all"):
        p = make_separation_figure()
        if p:
            written.append(p)

    if args.only in ("sao", "all"):
        p = make_sao_figure()
        if p:
            written.append(p)

    if args.only in ("downstream", "all"):
        written.append(make_downstream_figure())

    if args.only in ("chunk", "all"):
        p = make_chunk_layout_figure()
        if p:
            written.append(p)

    for path in written:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
