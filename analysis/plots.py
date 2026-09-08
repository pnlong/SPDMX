"""Plot song-length distributions for SA3 model selection."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from shared.config import SA3_MEDIUM_MAX_DURATION, SA3_SMALL_MUSIC_MAX_DURATION


def _savefig(
    fig: plt.Figure,
    output_path: str | Path,
    *,
    dpi: int = 150,
    pad_inches: float = 0.1,
) -> None:
    """Save a figure; PDFs use a transparent background for paper inclusion."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    kwargs: dict = {"dpi": dpi, "bbox_inches": "tight", "pad_inches": pad_inches}
    if output_path.suffix.lower() == ".pdf":
        kwargs.update(transparent=True, facecolor="none", edgecolor="none")
    fig.savefig(output_path, **kwargs)


def _add_sa3_limits(ax: plt.Axes):
    ax.axvline(
        SA3_SMALL_MUSIC_MAX_DURATION,
        color="C1",
        linestyle="--",
        linewidth=1.5,
        label=f"small-music ({SA3_SMALL_MUSIC_MAX_DURATION}s)",
    )
    ax.axvline(
        SA3_MEDIUM_MAX_DURATION,
        color="C2",
        linestyle="--",
        linewidth=1.5,
        label=f"medium ({SA3_MEDIUM_MAX_DURATION}s)",
    )


def plot_histogram(
    durations: pd.Series,
    output_path: str | Path,
    *,
    max_seconds: float = 600,
    bins: int = 60,
):
    """Histogram of song lengths with SA3 model duration limits marked."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    clipped = durations.clip(upper=max_seconds)
    ax.hist(clipped, bins=bins, color="C0", alpha=0.85, edgecolor="white")
    _add_sa3_limits(ax)
    ax.set_xlabel("Song length (seconds)")
    ax.set_ylabel("Count")
    ax.set_title("PDMX song length distribution")
    ax.set_xlim(0, max_seconds)
    ax.legend(loc="upper right")
    fig.tight_layout()
    _savefig(fig, output_path)
    plt.close(fig)


def plot_percentiles(
    durations: pd.Series,
    output_path: str | Path,
    *,
    max_seconds: float = 600,
):
    """Empirical CDF (percentile curve) with SA3 model duration limits marked."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sorted_durations = durations.sort_values().to_numpy()
    cumulative_pct = (pd.Series(range(1, len(sorted_durations) + 1)) / len(sorted_durations) * 100).to_numpy()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(sorted_durations, cumulative_pct, color="C0", linewidth=2)
    _add_sa3_limits(ax)
    ax.set_xlabel("Song length (seconds)")
    ax.set_ylabel("Percentile")
    ax.set_title("PDMX song length percentiles")
    ax.set_xlim(0, max_seconds)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    _savefig(fig, output_path)
    plt.close(fig)


def _gm_count_series(stems: pd.DataFrame) -> pd.Series:
    if stems is None or stems.empty:
        return pd.Series(dtype=int)
    return stems["gm_id"].value_counts()


def _select_gm_ids_for_plot(
    counts: pd.Series,
    *,
    top_n: int,
    drum_id: int,
) -> list:
    """Top-N gm_ids by count, keeping drums visible; optional Other via -1 elsewhere."""
    if counts.empty:
        return []
    if top_n <= 0 or len(counts) <= top_n:
        return list(counts.index)
    head_index = list(counts.head(top_n).index)
    if drum_id in counts.index and drum_id not in head_index:
        head_index = list(counts.head(top_n - 1).index)
        if drum_id not in head_index:
            head_index.append(drum_id)
    return head_index


def _count_labels_with_pct(
    values,
    total: int,
    *,
    min_pct: float = 1.0,
) -> list[str]:
    """Outside bar labels: ``12345`` or ``12345 (12%)`` when share exceeds ``min_pct``."""
    labels: list[str] = []
    for val in values:
        count = int(val)
        if total > 0 and count > 0:
            pct = 100.0 * float(count) / float(total)
            if pct > min_pct:
                labels.append(f"{count} ({round(pct)}%)")
                continue
        labels.append(f"{count}")
    return labels


def _style_gm_count_axis(ax) -> None:
    """Vertical gridlines behind bars for easier count reading."""
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, linestyle="--", linewidth=0.7, alpha=0.45, color="0.5")
    ax.yaxis.grid(False)


def plot_gm_program_bar(
    stems: pd.DataFrame,
    output_path: str | Path,
    *,
    top_n: int = 40,
    title: str = "PDMX General MIDI program usage (drums = channel 10)",
):
    """Horizontal bar chart of GM program id counts (top N + Other)."""
    from analysis.gm_programs import DRUM_GM_ID, gm_id_label

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    counts = _gm_count_series(stems)
    total = int(counts.sum())
    head_index = _select_gm_ids_for_plot(counts, top_n=top_n, drum_id=DRUM_GM_ID)
    if head_index:
        head = counts.loc[head_index]
        other = int(counts.drop(labels=head_index, errors="ignore").sum())
        plot_counts = head.copy()
        if other:
            plot_counts.loc[-1] = other
    else:
        plot_counts = counts

    plot_counts = plot_counts.sort_values(ascending=True)
    values = [int(v) for v in plot_counts.values]

    labels = [
        gm_id_label(int(v)) if int(v) >= 0 else "Other" for v in plot_counts.index
    ]

    fig_height = max(6, 0.28 * len(plot_counts))
    fig, ax = plt.subplots(figsize=(20, fig_height))
    bars = ax.barh(labels, values, color="C0", alpha=0.9)
    ax.bar_label(
        bars,
        labels=_count_labels_with_pct(values, total),
        padding=3,
        fontsize=8,
    )
    ax.set_xlim(0, max(values, default=0) * 1.22 + 1)
    _style_gm_count_axis(ax)
    ax.set_xlabel("Stem count (non-empty MIDI tracks)")
    ax.set_ylabel("GM program id")
    ax.set_title(title)
    fig.tight_layout()
    _savefig(fig, output_path)
    plt.close(fig)


def plot_gm_program_compare(
    stems_original: pd.DataFrame,
    stems_corrected: pd.DataFrame,
    output_path: str | Path,
    *,
    top_n: int = 10,
    rank_by: str = "corrected",
    show_percentages: bool = False,
    figsize: tuple[float, float] = (8.0, 4.0),
):
    """Grouped horizontal bar chart: original vs register-corrected GM usage.

    Selects the top ``top_n`` programs by ``rank_by`` (``corrected`` or
    ``original``) stem count and plots each program's share under both
    inventories. The long tail is omitted (no ``Other`` bucket). Rows are
    ordered by the ranking inventory (most → least). Default figsize is 2:1
    (wide). No figure title.
    """
    import seaborn as sns

    from analysis.gm_programs import gm_program_paper_label

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    xlabel = "Percentage of Stems (%)"
    if rank_by not in {"corrected", "original"}:
        raise ValueError("rank_by must be 'corrected' or 'original'")

    left_counts = _gm_count_series(stems_original)
    right_counts = _gm_count_series(stems_corrected)
    left_total = int(left_counts.sum())
    right_total = int(right_counts.sum())
    rank_counts = right_counts if rank_by == "corrected" else left_counts
    rank_total = right_total if rank_by == "corrected" else left_total

    if rank_counts.empty or rank_total <= 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_ylabel("General MIDI Program")
        ax.set_xlabel(xlabel)
        _savefig(fig, output_path)
        plt.close(fig)
        return

    if top_n <= 0 or len(rank_counts) <= top_n:
        ordered = list(rank_counts.sort_values(ascending=False).index)
    else:
        ordered = list(rank_counts.sort_values(ascending=False).head(top_n).index)

    # PDMX = raw MIDI program_change inventory; sPDMX = register-corrected inventory.
    hue_order = ["PDMX", "sPDMX"]
    rank_rows: list[tuple[str, float, float, float]] = []
    for gm_id in ordered:
        label = gm_program_paper_label(int(gm_id))
        left_n = int(left_counts.get(gm_id, 0))
        right_n = int(right_counts.get(gm_id, 0))
        left_pct = 100.0 * left_n / left_total if left_total else 0.0
        right_pct = 100.0 * right_n / right_total if right_total else 0.0
        rank_pct = right_pct if rank_by == "corrected" else left_pct
        rank_rows.append((label, left_pct, right_pct, rank_pct))

    rank_rows.sort(key=lambda row: (-row[3], row[0]))
    labels = [label for label, _, _, _ in rank_rows]

    rows: list[dict] = []
    for label, left_pct, right_pct, _rank_pct in rank_rows:
        rows.append({"label": label, "source": "PDMX", "pct": left_pct})
        rows.append({"label": label, "source": "sPDMX", "pct": right_pct})
    plot_df = pd.DataFrame(rows)

    sns.set_theme(style="ticks", context="paper")
    try:
        fig, ax = plt.subplots(figsize=figsize)
        sns.barplot(
            data=plot_df,
            y="label",
            x="pct",
            hue="source",
            order=labels,
            hue_order=hue_order,
            orient="h",
            ax=ax,
            palette={"PDMX": "C0", "sPDMX": "C1"},
            saturation=0.9,
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("General MIDI Program")
        ax.set_title("")
        ax.legend(loc="lower right", frameon=True, fontsize=8, title=None)
        ax.set_xlim(0, max(float(plot_df["pct"].max()) * 1.12, 1.0))
        _style_gm_count_axis(ax)
        sns.despine(ax=ax)
        ax.tick_params(axis="y", labelsize=8)
        ax.tick_params(axis="x", labelsize=8)

        if show_percentages:
            for container in ax.containers:
                ax.bar_label(container, fmt="%.0f%%", padding=2, fontsize=7)

        fig.tight_layout()
        _savefig(fig, output_path, pad_inches=0.02)
        plt.close(fig)
    finally:
        sns.reset_defaults()


def plot_track_name_bar(
    stems: pd.DataFrame,
    output_path: str | Path,
    *,
    top_n: int = 40,
):
    """Horizontal bar chart of track name counts (top N + Other)."""
    from analysis.track_names import UNNAMED_TRACK

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if stems.empty:
        counts = pd.Series(dtype=int)
    else:
        counts = stems["track_name"].value_counts()

    if top_n > 0 and len(counts) > top_n:
        head = counts.head(top_n)
        other = int(counts.iloc[top_n:].sum())
        plot_counts = head.copy()
        if other:
            plot_counts.loc["__other__"] = other
    else:
        plot_counts = counts

    plot_counts = plot_counts.sort_values(ascending=True)
    labels = [
        name if name != "__other__" else "Other"
        for name in plot_counts.index
    ]

    fig_height = max(6, 0.28 * len(plot_counts))
    fig, ax = plt.subplots(figsize=(12, fig_height))
    bars = ax.barh(labels, plot_counts.values, color="C0", alpha=0.9)
    ax.bar_label(bars, fmt="%d", padding=3, fontsize=8)
    ax.set_xlabel("Track count (non-empty MIDI tracks)")
    ax.set_ylabel("Track name")
    ax.set_title("PDMX MIDI track name usage")
    if UNNAMED_TRACK in labels:
        idx = labels.index(UNNAMED_TRACK)
        bars[idx].set_color("C3")
    fig.tight_layout()
    _savefig(fig, output_path)
    plt.close(fig)


_ARM_LABELS = {
    "slakh": "Slakh",
    "spdmx_matched": "sPDMX-matched",
    "spdmx_full": "sPDMX-full",
    "Slakh": "Slakh",
    "sPDMX-matched": "sPDMX-matched",
    "sPDMX-full": "sPDMX-full",
}

_ARM_ORDER = ("Slakh", "sPDMX-matched", "sPDMX-full")
_TARGET_ORDER = ("bass", "drums", "guitar", "piano")
# Non-realify ablation arms reported in the ICASSP draft (SA3 omitted).
_CONDITION_ORDER = ("A1", "B1", "CA1", "CB1")


def _annotate_bars(ax, fmt: str = "{:.1f}", fontsize: int = 7) -> None:
    for container in ax.containers:
        labels = []
        for patch in container:
            h = patch.get_height()
            if h != h:  # NaN
                labels.append("")
            else:
                labels.append(fmt.format(h))
        ax.bar_label(container, labels=labels, padding=2, fontsize=fontsize)


def plot_ablation_listening(
    scores: pd.DataFrame,
    output_path: str | Path,
    *,
    figsize: tuple[float, float] = (8.0, 3.6),
) -> None:
    """Grouped bars: condition × {content, realism} with value labels.

    Expects columns: ``condition``, ``content``, ``realism``.
    """
    import seaborn as sns

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = scores.copy()
    df["condition"] = pd.Categorical(df["condition"], categories=list(_CONDITION_ORDER), ordered=True)
    long = df.melt(
        id_vars=["condition"],
        value_vars=["content", "realism"],
        var_name="scale",
        value_name="score",
    )
    long["scale"] = long["scale"].str.capitalize()

    sns.set_theme(style="ticks", context="paper")
    try:
        fig, ax = plt.subplots(figsize=figsize)
        sns.barplot(
            data=long,
            x="condition",
            y="score",
            hue="scale",
            order=list(_CONDITION_ORDER),
            hue_order=["Content", "Realism"],
            ax=ax,
            saturation=0.9,
        )
        _annotate_bars(ax, fmt="{:.1f}")
        ax.set_xlabel("Condition")
        ax.set_ylabel("Mean score (0–100)")
        ax.set_ylim(0, 100)
        ax.legend(title=None, frameon=True, fontsize=8)
        sns.despine(ax=ax)
        fig.tight_layout()
        _savefig(fig, output_path, pad_inches=0.02)
        plt.close(fig)
    finally:
        sns.reset_defaults()


def plot_separation_sisdr(
    summary: pd.DataFrame,
    output_path: str | Path,
    *,
    figsize: tuple[float, float] = (8.0, 4.2),
) -> None:
    """SI-SDR bars: instrument × train arm, optional MUSDB panel.

    Expects columns: ``train_arm``, ``test_set``, ``target``, ``si_sdr_mean``.
    """
    import seaborn as sns

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = summary.copy()
    df["train_arm"] = df["train_arm"].map(lambda a: _ARM_LABELS.get(str(a), str(a)))
    df["target"] = df["target"].str.lower()
    test_sets = [t for t in ("slakh2100", "musdb18") if t in set(df["test_set"].astype(str))]
    if not test_sets:
        test_sets = sorted(df["test_set"].astype(str).unique())

    sns.set_theme(style="ticks", context="paper")
    try:
        n = len(test_sets)
        fig, axes = plt.subplots(1, n, figsize=figsize, sharey=True)
        if n == 1:
            axes = [axes]
        for ax, test_set in zip(axes, test_sets):
            sub = df[df["test_set"].astype(str) == test_set]
            targets = [t for t in _TARGET_ORDER if t in set(sub["target"])]
            if test_set == "musdb18":
                targets = [t for t in ("bass", "drums") if t in set(sub["target"])]
            sns.barplot(
                data=sub,
                x="target",
                y="si_sdr_mean",
                hue="train_arm",
                order=targets,
                hue_order=[a for a in _ARM_ORDER if a in set(sub["train_arm"])],
                ax=ax,
                saturation=0.9,
            )
            _annotate_bars(ax, fmt="{:.1f}")
            title = "Slakh2100 test" if test_set == "slakh2100" else "MUSDB18 test"
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("Target")
            ax.set_ylabel("SI-SDR (dB)" if ax is axes[0] else "")
            ax.legend(title=None, frameon=True, fontsize=7)
            sns.despine(ax=ax)
        fig.tight_layout()
        _savefig(fig, output_path, pad_inches=0.02)
        plt.close(fig)
    finally:
        sns.reset_defaults()


def plot_sao_metrics(
    metrics: pd.DataFrame,
    output_path: str | Path,
    *,
    figsize: tuple[float, float] = (7.0, 3.4),
) -> None:
    """Two-panel FAD (lower better) and CLAP (higher better) by train arm."""
    import seaborn as sns

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = metrics.copy()
    df["train_arm"] = df["train_arm"].map(lambda a: _ARM_LABELS.get(str(a), str(a)))
    arm_order = [a for a in _ARM_ORDER if a in set(df["train_arm"])]

    sns.set_theme(style="ticks", context="paper")
    try:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        for ax, col, ylabel, title in (
            (axes[0], "fad", "FAD ↓", "FAD"),
            (axes[1], "clap", "CLAP ↑", "CLAP"),
        ):
            sns.barplot(
                data=df,
                x="train_arm",
                y=col,
                order=arm_order,
                ax=ax,
                color="C0",
                saturation=0.9,
            )
            _annotate_bars(ax, fmt="{:.3g}")
            ax.set_xlabel("Training data")
            ax.set_ylabel(ylabel)
            ax.set_title(title, fontsize=10)
            ax.tick_params(axis="x", rotation=15)
            sns.despine(ax=ax)
        fig.tight_layout()
        _savefig(fig, output_path, pad_inches=0.02)
        plt.close(fig)
    finally:
        sns.reset_defaults()


def plot_downstream_poc(
    separation: pd.DataFrame,
    sao: pd.DataFrame,
    output_path: str | Path,
    *,
    figsize: tuple[float, float] = (8.5, 3.2),
) -> None:
    """Single figure: SI-SDR (Slakh test) | FAD | CLAP for the three train arms."""
    import seaborn as sns

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sep = separation.copy()
    sep["train_arm"] = sep["train_arm"].map(lambda a: _ARM_LABELS.get(str(a), str(a)))
    sep["target"] = sep["target"].astype(str).str.lower()
    sep = sep[sep["test_set"].astype(str) == "slakh2100"]
    sao_df = sao.copy()
    sao_df["train_arm"] = sao_df["train_arm"].map(lambda a: _ARM_LABELS.get(str(a), str(a)))
    arm_order = [a for a in _ARM_ORDER if a in set(sep["train_arm"]) | set(sao_df["train_arm"])]
    targets = [t for t in _TARGET_ORDER if t in set(sep["target"])]

    sns.set_theme(style="ticks", context="paper")
    try:
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        # Panel 0: SI-SDR
        if not sep.empty and sep["si_sdr_mean"].notna().any():
            sns.barplot(
                data=sep,
                x="target",
                y="si_sdr_mean",
                hue="train_arm",
                order=targets,
                hue_order=arm_order,
                ax=axes[0],
                saturation=0.9,
            )
            _annotate_bars(axes[0], fmt="{:.1f}")
            axes[0].legend(title=None, frameon=True, fontsize=6)
        else:
            axes[0].text(0.5, 0.5, "SI-SDR pending", ha="center", va="center")
            axes[0].set_xticks([])
            axes[0].set_yticks([])
        axes[0].set_title("Demucs SI-SDR", fontsize=10)
        axes[0].set_xlabel("Target")
        axes[0].set_ylabel("SI-SDR (dB)")

        for ax, col, ylabel, title in (
            (axes[1], "fad", "FAD ↓", "SAO FAD"),
            (axes[2], "clap", "CLAP ↑", "SAO CLAP"),
        ):
            if not sao_df.empty and sao_df[col].notna().any():
                sns.barplot(
                    data=sao_df,
                    x="train_arm",
                    y=col,
                    order=arm_order,
                    ax=ax,
                    color="C0",
                    saturation=0.9,
                )
                _annotate_bars(ax, fmt="{:.3g}")
            else:
                ax.text(0.5, 0.5, f"{title} pending", ha="center", va="center")
                ax.set_xticks([])
                ax.set_yticks([])
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("Training data")
            ax.set_ylabel(ylabel)
            ax.tick_params(axis="x", rotation=15)
            sns.despine(ax=ax)
        sns.despine(ax=axes[0])
        fig.tight_layout()
        _savefig(fig, output_path, pad_inches=0.02)
        plt.close(fig)
    finally:
        sns.reset_defaults()
