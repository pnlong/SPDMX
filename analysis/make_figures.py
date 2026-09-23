"""Regenerate ICASSP paper figures as transparent PDFs under submission/figs/."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from analysis.gm_programs import DRUM_GM_ID, gm_program_paper_label
from analysis.plots import (
    INSTRUMENT_CATEGORY_YLABEL,
    _savefig,
    listening_category_label,
    plot_ablation_listening,
    plot_ablation_listening_panels,
    plot_chunk_layout,
    plot_downstream_poc,
    plot_gm_stems_vs_hours,
    plot_sao_metrics,
    plot_separation_multistem,
    plot_separation_sisdr,
    plot_song_hours_by_stems,
    plot_stems_per_song,
)
from shared.config import OUTPUT_DIR, SPDMX_DEV_DIR_NAME, SPDMX_FILE_NAME
from synthesis.chunking import CHUNKS_FILE_NAME
from synthesis.patches import LISTENING_CATEGORY_GM_CLASSES, resolve_probe_category
from synthesis.recipe import METHOD_MIDI_DDSP, load_recipe

REPO_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = REPO_ROOT / "submission" / "figs"
DATA_DIR = REPO_ROOT / "analysis" / "paper_data"
SEP_PAPER_CSV = (
    Path(OUTPUT_DIR) / "dev" / "experiments" / "separation" / "eval" / "separation_sisdr.csv"
)
SEP_MULTISTEM_CSV = (
    Path(OUTPUT_DIR)
    / "dev"
    / "experiments"
    / "separation"
    / "eval"
    / "separation_sisdr_multistem.csv"
)
SAO_PAPER_CSV = (
    Path(OUTPUT_DIR) / "dev" / "experiments" / "sao" / "metrics" / "sao_metrics.csv"
)
INSTRUMENTS_DIR = (
    Path(OUTPUT_DIR) / "dev" / "analysis" / "instruments" / "all_valid"
)
ORIGINAL_STEMS = INSTRUMENTS_DIR / "gm_program_stems.csv"
CORRECTED_STEMS = INSTRUMENTS_DIR / "gm_program_stems_corrected.csv"
CHUNKS_CSV = Path(OUTPUT_DIR) / "SPDMX" / CHUNKS_FILE_NAME
TRACKS_PER_SONG_JSON = DATA_DIR / "tracks_per_song.json"
TRACKS_PER_SONG_DOCS = REPO_ROOT / "docs" / "data" / "tracks_per_song.json"
SONG_HOURS_BY_STEMS_JSON = DATA_DIR / "song_hours_by_stems.json"
SONG_HOURS_BY_STEMS_DOCS = REPO_ROOT / "docs" / "data" / "song_hours_by_stems.json"
SPDMX_DEV_STEMS = Path("/deepfreeze/share/SPDMX") / SPDMX_DEV_DIR_NAME / f"{SPDMX_FILE_NAME}.csv"
SPDMX_DEV_STEMS_FALLBACK = Path(OUTPUT_DIR) / SPDMX_DEV_DIR_NAME / f"{SPDMX_FILE_NAME}.csv"
ACTIVE_HOURS_CSV = REPO_ROOT / "analysis" / "output" / "active_hours" / "program_active_hours.csv"
ACTIVE_HOURS_CSV_DATA = DATA_DIR / "program_active_hours.csv"


def _resolve_paper_csv(preferred: Path, fallback_name: str) -> Path:
    """Prefer deepfreeze experiment outputs; fall back to analysis/paper_data/."""
    if preferred.is_file():
        return preferred
    return DATA_DIR / fallback_name


def _spdmx_program_hours_summary(stems_csv: Path) -> pd.DataFrame:
    stems = pd.read_csv(stems_csv, usecols=["program", "is_drum", "song_length"])
    gm_ids = [
        DRUM_GM_ID if bool(is_drum) else int(program)
        for program, is_drum in zip(stems["program"], stems["is_drum"])
    ]
    stems = stems.assign(gm_id=gm_ids)
    grouped = (
        stems.groupby("gm_id", sort=False)
        .agg(n_stems=("song_length", "size"), wall_seconds=("song_length", "sum"))
        .reset_index()
    )
    grouped["wall_hours"] = grouped["wall_seconds"] / 3600.0
    grouped["label"] = grouped["gm_id"].map(lambda g: gm_program_paper_label(int(g)))
    return grouped


def _attach_active_hours(summary: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """Join RMS-active hours when ``program_active_hours.csv`` is available."""
    active_path = ACTIVE_HOURS_CSV if ACTIVE_HOURS_CSV.is_file() else ACTIVE_HOURS_CSV_DATA
    if not active_path.is_file():
        return summary, False
    active = pd.read_csv(active_path, usecols=["gm_id", "active_hours", "n_stems"])
    n_measured = int(active["n_stems"].sum())
    merged = summary.drop(columns=["active_hours"], errors="ignore").merge(
        active[["gm_id", "active_hours"]],
        on="gm_id",
        how="left",
    )
    merged["active_hours"] = merged["active_hours"].fillna(0.0)
    print(
        f"using RMS-active hours from {active_path} "
        f"({n_measured:,} measured stems)",
        flush=True,
    )
    return merged, True


def _category_display_label(category: str) -> str:
    return listening_category_label(category)


def _ddsp_eligible_categories() -> set[str]:
    recipe = load_recipe()
    return {
        category
        for category, spec in recipe.specs.items()
        if spec.method == METHOD_MIDI_DDSP
    }


def _spdmx_category_hours_summary(stems_csv: Path) -> pd.DataFrame:
    """Aggregate stems and hours by the 10 listening categories."""
    usecols = ["program", "is_drum", "song_length"]
    header = pd.read_csv(stems_csv, nrows=0).columns.tolist()
    if "name" in header:
        usecols = ["program", "is_drum", "name", "song_length"]
    stems = pd.read_csv(stems_csv, usecols=usecols)
    names = (
        stems["name"].tolist()
        if "name" in stems.columns
        else [None] * len(stems)
    )
    categories = [
        resolve_probe_category(
            program=int(program),
            is_drum=bool(is_drum),
            track_name=None if (not isinstance(name, str) or not name.strip()) else name,
        )
        for program, is_drum, name in zip(stems["program"], stems["is_drum"], names)
    ]
    gm_ids = [
        DRUM_GM_ID if bool(is_drum) else int(program)
        for program, is_drum in zip(stems["program"], stems["is_drum"])
    ]
    stems = stems.assign(category=categories, gm_id=gm_ids)
    grouped = (
        stems.groupby("category", sort=False)
        .agg(n_stems=("song_length", "size"), wall_seconds=("song_length", "sum"))
        .reset_index()
    )
    grouped["wall_hours"] = grouped["wall_seconds"] / 3600.0

    # Attribute RMS-active hours from the per-program table onto categories.
    active_path = ACTIVE_HOURS_CSV if ACTIVE_HOURS_CSV.is_file() else ACTIVE_HOURS_CSV_DATA
    if active_path.is_file():
        active = pd.read_csv(active_path, usecols=["gm_id", "active_hours", "n_stems"])
        active = active[active["n_stems"] > 0].copy()
        active["hours_per_stem"] = active["active_hours"] / active["n_stems"]
        stems = stems.merge(active[["gm_id", "hours_per_stem"]], on="gm_id", how="left")
        stems["hours_per_stem"] = stems["hours_per_stem"].fillna(0.0)
        active_by_cat = (
            stems.groupby("category", sort=False)["hours_per_stem"].sum().rename("active_hours")
        )
        grouped = grouped.merge(active_by_cat, on="category", how="left")
        grouped["active_hours"] = grouped["active_hours"].fillna(0.0)
        print(
            f"using RMS-active hours from {active_path} "
            f"({int(active['n_stems'].sum()):,} measured stems)",
            flush=True,
        )

    order = list(LISTENING_CATEGORY_GM_CLASSES.keys())
    grouped = (
        pd.DataFrame({"category": order})
        .merge(grouped, on="category", how="left")
        .fillna({"n_stems": 0, "wall_seconds": 0.0, "wall_hours": 0.0})
    )
    if "active_hours" in grouped.columns:
        grouped["active_hours"] = grouped["active_hours"].fillna(0.0)
    ddsp = _ddsp_eligible_categories()
    grouped["ddsp_eligible"] = grouped["category"].isin(ddsp)
    grouped["label"] = grouped["category"].map(_category_display_label)
    return grouped


def make_gm_program_compare_figure(
    *,
    top_n: int | None = None,
    rank_by: str = "stems",
) -> Path:
    """Stem-count vs hours by listening category (DDSP-eligible marked with *)."""
    stems_csv = SPDMX_DEV_STEMS if SPDMX_DEV_STEMS.is_file() else SPDMX_DEV_STEMS_FALLBACK
    if not stems_csv.is_file():
        raise FileNotFoundError(f"missing SPDMX stems table: {stems_csv}")
    summary = _spdmx_category_hours_summary(stems_csv)
    hours_col = "active_hours" if "active_hours" in summary.columns else "wall_hours"
    out = FIGURES_DIR / "gm_program_counts_compare.pdf"
    plot_gm_stems_vs_hours(
        summary,
        out,
        top_n=top_n,
        rank_by=rank_by,
        hours_col=hours_col,
        hours_title="Hours",
        ylabel=INSTRUMENT_CATEGORY_YLABEL,
        figsize=(7.0, 3.4),
    )
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    summary.sort_values("n_stems", ascending=False).to_csv(
        DATA_DIR / "category_stem_hours.csv", index=False
    )
    # Keep legacy program-level CSV for other tooling when still useful.
    program_summary = _spdmx_program_hours_summary(stems_csv)
    program_summary, has_active = _attach_active_hours(program_summary)
    program_summary.sort_values("n_stems", ascending=False).to_csv(
        DATA_DIR / "program_stem_hours.csv", index=False
    )
    if has_active:
        ACTIVE_HOURS_CSV_DATA.parent.mkdir(parents=True, exist_ok=True)
        src = ACTIVE_HOURS_CSV if ACTIVE_HOURS_CSV.is_file() else ACTIVE_HOURS_CSV_DATA
        if src.is_file() and src.resolve() != ACTIVE_HOURS_CSV_DATA.resolve():
            ACTIVE_HOURS_CSV_DATA.write_text(src.read_text())
    return out


def make_stems_per_song_figure() -> Path | None:
    """Song-hours vs stems/song (SPDMX vs Slakh); falls back to legacy counts."""
    hours_path = (
        SONG_HOURS_BY_STEMS_JSON
        if SONG_HOURS_BY_STEMS_JSON.is_file()
        else SONG_HOURS_BY_STEMS_DOCS
    )
    out = FIGURES_DIR / "stems_per_song.pdf"
    if hours_path.is_file():
        payload = json.loads(hours_path.read_text(encoding="utf-8"))
        plot_song_hours_by_stems(payload, out, figsize=(3.45, 2.45), log_y=True)
        return out
    path = TRACKS_PER_SONG_JSON if TRACKS_PER_SONG_JSON.is_file() else TRACKS_PER_SONG_DOCS
    if not path.is_file():
        print(
            f"skip stems-per-song figure: missing {hours_path} and count hist "
            f"({TRACKS_PER_SONG_JSON} / {TRACKS_PER_SONG_DOCS})"
        )
        return None
    hist = json.loads(path.read_text(encoding="utf-8"))
    plot_stems_per_song(hist, out, figsize=(3.45, 2.35))
    return out


def make_ablation_listening_figure() -> Path | None:
    scores_path = DATA_DIR / "ablation_listening_scores.csv"
    by_cat_path = DATA_DIR / "ablation_listening_by_category.csv"
    summary_path = DATA_DIR / "ablation_listening.csv"
    out = FIGURES_DIR / "ablation_listening.pdf"
    if by_cat_path.is_file():
        plot_ablation_listening_panels(
            pd.read_csv(by_cat_path),
            out,
            layout="grid",
            figsize=(9.0, 3.2),
        )
        return out
    if scores_path.is_file():
        plot_ablation_listening(
            pd.read_csv(scores_path),
            out,
            figsize=(5.5, 3.2),
            scales=("realism",),
        )
        return out
    if not summary_path.is_file():
        print(
            f"skip ablation figure: missing {by_cat_path}, {scores_path}, and {summary_path}"
        )
        return None
    plot_ablation_listening(pd.read_csv(summary_path), out, figsize=(5.5, 3.2))
    return out


def make_separation_figure() -> list[Path]:
    """Write LaTeX panels (a) BDGP comparison and (b) multistem."""
    written: list[Path] = []
    csv_path = _resolve_paper_csv(SEP_PAPER_CSV, "separation_sisdr.csv")
    out_a = FIGURES_DIR / "separation_sisdr.pdf"
    if not csv_path.is_file():
        _placeholder_figure(out_a, "SI-SDR results pending")
    else:
        df = pd.read_csv(csv_path)
        if df.empty or df["si_sdr_mean"].isna().all():
            _placeholder_figure(out_a, "SI-SDR results pending")
        else:
            plot_separation_sisdr(df, out_a)
    written.append(out_a)

    ms_path = _resolve_paper_csv(SEP_MULTISTEM_CSV, "separation_sisdr_multistem.csv")
    out_b = FIGURES_DIR / "separation_sisdr_multistem.pdf"
    skinny = (1.9, 3.6)
    if not ms_path.is_file():
        _placeholder_figure(out_b, "pending", figsize=skinny)
    else:
        ms = pd.read_csv(ms_path)
        if ms.empty or ms["si_sdr_mean"].isna().all():
            _placeholder_figure(out_b, "pending", figsize=skinny)
        else:
            plot_separation_multistem(ms, out_b)
    written.append(out_b)
    return written


def make_sao_figure() -> Path | None:
    csv_path = _resolve_paper_csv(SAO_PAPER_CSV, "sao_metrics.csv")
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
    sep_path = _resolve_paper_csv(SEP_PAPER_CSV, "separation_sisdr.csv")
    sao_path = _resolve_paper_csv(SAO_PAPER_CSV, "sao_metrics.csv")
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


def _placeholder_figure(
    path: Path,
    message: str,
    *,
    figsize: tuple[float, float] = (6.0, 2.5),
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=10)
    _savefig(fig, path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-n", type=int, default=None)
    parser.add_argument(
        "--rank-by",
        choices=("stems", "hours"),
        default="stems",
        help="Rank listening categories by stem count or hours.",
    )
    parser.add_argument(
        "--only",
        choices=(
            "gm",
            "stems",
            "ablation",
            "separation",
            "sao",
            "downstream",
            "chunk",
            "all",
        ),
        default="all",
    )
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    if args.only in ("gm", "all"):
        stems_csv = SPDMX_DEV_STEMS if SPDMX_DEV_STEMS.is_file() else SPDMX_DEV_STEMS_FALLBACK
        if stems_csv.is_file():
            written.append(
                make_gm_program_compare_figure(
                    top_n=args.top_n,
                    rank_by=args.rank_by,
                )
            )
        else:
            print(f"skip GM figure: missing {stems_csv}")

    if args.only in ("stems", "all"):
        p = make_stems_per_song_figure()
        if p:
            written.append(p)

    if args.only in ("ablation", "all"):
        p = make_ablation_listening_figure()
        if p:
            written.append(p)

    if args.only in ("separation", "all"):
        written.extend(make_separation_figure())

    if args.only in ("sao", "all"):
        p = make_sao_figure()
        if p:
            written.append(p)

    if args.only == "downstream":
        written.append(make_downstream_figure())

    if args.only in ("chunk", "all"):
        p = make_chunk_layout_figure()
        if p:
            written.append(p)

    for path in written:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
