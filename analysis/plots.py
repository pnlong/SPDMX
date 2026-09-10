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
    """Save a figure with a transparent background (PDF/PNG/SVG)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    kwargs: dict = {
        "dpi": dpi,
        "bbox_inches": "tight",
        "pad_inches": pad_inches,
        "transparent": True,
        "facecolor": "none",
        "edgecolor": "none",
    }
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
    "spdmx": "sPDMX",
    "spdmx_matched": "sPDMX-BDGP",
    "spdmx_full": "sPDMX-full",
    "Slakh": "Slakh",
    "sPDMX": "sPDMX",
    "sPDMX-matched": "sPDMX-BDGP",
    "sPDMX-BDGP": "sPDMX-BDGP",
    "sPDMX-full": "sPDMX-full",
}

_ARM_ORDER_SEP = ("Slakh", "sPDMX")
_ARM_ORDER_SAO = ("Slakh", "sPDMX-BDGP", "sPDMX-full")
_ARM_ORDER = _ARM_ORDER_SAO  # default for combined/legacy callers
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
                hue_order=[a for a in _ARM_ORDER_SEP if a in set(sub["train_arm"])],
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
    arm_order = [a for a in _ARM_ORDER_SAO if a in set(df["train_arm"])]

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


def plot_chunk_layout(
    chunks: pd.DataFrame,
    output_path: str | Path,
    *,
    target_bytes: int | None = None,
    figsize: tuple[float, float] = (13.0, 5.3),
    n_chunk_preview: int = 7,
) -> None:
    """Wide schematic of the SPDMX release tree (metadata → chunks → song files)."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, PathPatch
    from matplotlib.path import Path as MplPath

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = chunks.copy()
    df["chunk"] = df["chunk"].map(lambda c: int(str(c).strip()))
    n_chunks = len(df)
    if target_bytes is None and "bytes" in df.columns and len(df):
        target_bytes = int(df["bytes"].mean())
    if target_bytes is None:
        from synthesis.chunking import CHUNK_BYTES_TARGET

        target_bytes = CHUNK_BYTES_TARGET
    target_gib = target_bytes / float(1024**3)

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_alpha(0.0)
    ax.set_xlim(0, 32)
    ax.set_ylim(0, 13.2)
    ax.axis("off")
    ax.set_facecolor("none")

    mono = "DejaVu Sans Mono"
    ink = "#1a1510"
    muted = "#6a5a48"

    def _frame(x: float, y: float, w: float, h: float, *, fc: str, ec: str = ink):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.012,rounding_size=0.16",
                linewidth=1.1,
                facecolor=fc,
                edgecolor=ec,
            )
        )

    def _caption(x: float, y: float, text: str, *, fontsize: float = 7.5):
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontstyle="italic",
            color=muted,
        )

    def _title_box(
        x: float,
        y: float,
        w: float,
        h: float,
        title: str,
        *,
        fc: str,
        ec: str = ink,
        fontsize: float = 9,
        family: str | None = None,
        weight: str = "bold",
        color: str = ink,
    ):
        _frame(x, y, w, h, fc=fc, ec=ec)
        ax.text(
            x + w / 2,
            y + h / 2,
            title,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontfamily=family,
            fontweight=weight,
            color=color,
            clip_on=True,
        )

    def _meta_box(
        x: float,
        y: float,
        w: float,
        h: float,
        filename: str,
        subtitle: str,
        *,
        fc: str,
    ):
        _frame(x, y, w, h, fc=fc)
        cy = y + h / 2
        ax.text(
            x + w / 2,
            cy + 0.2,
            filename,
            ha="center",
            va="center",
            fontsize=9,
            fontfamily=mono,
            fontweight="bold",
            color=ink,
            clip_on=True,
        )
        ax.text(
            x + w / 2,
            cy - 0.28,
            subtitle,
            ha="center",
            va="center",
            fontsize=7.5,
            fontstyle="italic",
            color=muted,
            clip_on=True,
        )

    def _chunk_box(
        x: float,
        y: float,
        w: float,
        h: float,
        name: str | None,
        size: str | None,
        *,
        fc: str,
    ):
        _frame(x, y, w, h, fc=fc)
        if name is None:
            ax.text(
                x + w / 2,
                y + h / 2,
                "···",
                ha="center",
                va="center",
                fontsize=10,
                color=ink,
            )
            return
        cy = y + h / 2
        ax.text(
            x + w / 2,
            cy + 0.18,
            name,
            ha="center",
            va="center",
            fontsize=7.5,
            fontfamily=mono,
            fontweight="bold",
            color=ink,
            clip_on=True,
        )
        if size:
            ax.text(
                x + w / 2,
                cy - 0.28,
                size,
                ha="center",
                va="center",
                fontsize=7,
                fontstyle="italic",
                color=muted,
                clip_on=True,
            )

    def _arrow(x1: float, y1: float, x2: float, y2: float):
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="-|>",
                mutation_scale=11,
                linewidth=1.1,
                color=muted,
            )
        )

    def _curly_brace(x: float, y0: float, y1: float, *, tip: float = 0.55):
        """Right-opening brace spanning [y0, y1] with spine near x."""
        mid = 0.5 * (y0 + y1)
        tip_x = x + tip
        spine = x + tip * 0.18
        cusp = x
        verts = [
            (tip_x, y1),
            (spine, y1),
            (spine, mid + (y1 - mid) * 0.55),
            (cusp, mid),
            (spine, mid - (mid - y0) * 0.55),
            (spine, y0),
            (tip_x, y0),
        ]
        codes = [
            MplPath.MOVETO,
            MplPath.CURVE3,
            MplPath.CURVE3,
            MplPath.CURVE3,
            MplPath.CURVE3,
            MplPath.CURVE3,
            MplPath.CURVE3,
        ]
        ax.add_patch(
            PathPatch(
                MplPath(verts, codes),
                facecolor="none",
                edgecolor=ink,
                linewidth=2.0,
                joinstyle="round",
                capstyle="round",
            )
        )

    # Row 1: root
    _title_box(
        10.5,
        11.55,
        11.0,
        1.05,
        "SPDMX release",
        fc="#e8922e",
        ec="#e8922e",
        fontsize=13,
        weight="bold",
    )
    _arrow(16.0, 11.55, 16.0, 10.85)
    _caption(16.0, 10.35, "filter CSVs first, then download only the media you need")
    _arrow(16.0, 9.85, 16.0, 9.15)

    # Row 2: metadata CSVs
    csv_w, csv_h, csv_y = 8.4, 1.15, 7.85
    for x, name, sub in (
        (1.8, "stems.csv", "one row per stem"),
        (11.8, "songs.csv", "one row per song"),
        (21.8, "chunks.csv", "archive index"),
    ):
        _meta_box(x, csv_y, csv_w, csv_h, name, sub, fc="#ffe2b8")
    _arrow(16.0, 7.85, 16.0, 7.0)

    # Row 3: chunk archives
    preview = min(n_chunk_preview, max(n_chunks, 1))
    gap = 0.3
    total_w = 28.4
    box_w = (total_w - gap * (preview - 1)) / preview
    chunk_y, chunk_h = 5.7, 1.15
    size_lbl = f"~{target_gib:.0f} GiB"
    zoom_center_x = 1.8 + box_w / 2
    for i in range(preview):
        x = 1.8 + i * (box_w + gap)
        if n_chunks <= preview:
            _chunk_box(x, chunk_y, box_w, chunk_h, f"chunk_{i}", size_lbl, fc="#f0b068")
        elif i < preview - 2:
            _chunk_box(x, chunk_y, box_w, chunk_h, f"chunk_{i}", size_lbl, fc="#f0b068")
        elif i == preview - 2:
            _chunk_box(x, chunk_y, box_w, chunk_h, None, None, fc="#f0b068")
        else:
            _chunk_box(
                x,
                chunk_y,
                box_w,
                chunk_h,
                f"chunk_{n_chunks - 1}",
                size_lbl,
                fc="#f0b068",
            )
        if i == 0:
            zoom_center_x = x + box_w / 2
    _caption(16.0, 5.2, f"{n_chunks} zip archives · ≈ {target_gib:.0f} GiB each")
    _arrow(zoom_center_x, 4.7, zoom_center_x, 3.95)

    # Bottom: song directory → MIDI derives stems; stems sum to mix
    song_x, song_y, song_w, song_h = 1.8, 1.15, 7.0, 2.5
    _title_box(
        song_x,
        song_y,
        song_w,
        song_h,
        "chunk_N/<song_id>/",
        fc="#ffd9a0",
        fontsize=11,
        family=mono,
        weight="bold",
    )

    brace_x = song_x + song_w + 0.35
    _curly_brace(brace_x, song_y + 0.15, song_y + song_h - 0.15, tip=0.5)

    cy = song_y + song_h / 2
    box_h = 0.78
    box_y = cy - box_h / 2
    x = brace_x + 0.85

    def _name_box(x0: float, w: float, name: str, *, fc: str, fontsize: float = 8):
        _frame(x0, box_y, w, box_h, fc=fc)
        ax.text(
            x0 + w / 2,
            cy,
            name,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontfamily=mono,
            fontweight="bold",
            color=ink,
            clip_on=True,
        )

    def _op(x0: float, glyph: str, *, fontsize: float = 12) -> float:
        ax.text(
            x0,
            cy,
            glyph,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight="bold",
            color=ink,
        )
        return x0

    # mix.mid is the source
    mid_w = 3.4
    _name_box(x, mid_w, "mix.mid", fc="#e8c9a0", fontsize=8.5)
    ax.text(
        x + mid_w / 2,
        box_y - 0.28,
        "paired MIDI",
        ha="center",
        va="top",
        fontsize=7,
        fontstyle="italic",
        color=muted,
    )
    x += mid_w + 0.35
    _arrow(x - 0.28, cy, x + 0.15, cy)
    x += 0.35

    # stems combined with +
    stem_w = 2.15
    stems = ("0.flac", "1.flac", "···", "K.flac")
    stem_left = x
    for i, name in enumerate(stems):
        if name == "···":
            _frame(x, box_y, stem_w * 0.7, box_h, fc="#fff6ea")
            ax.text(
                x + stem_w * 0.35,
                cy,
                "···",
                ha="center",
                va="center",
                fontsize=10,
                color=ink,
            )
            x += stem_w * 0.7
        else:
            _name_box(x, stem_w, name, fc="#fff6ea", fontsize=7.5)
            x += stem_w
        if i < len(stems) - 1:
            x += 0.28
            _op(x, "+")
            x += 0.28
    stem_right = x
    ax.text(
        0.5 * (stem_left + stem_right),
        box_y - 0.28,
        "stems",
        ha="center",
        va="top",
        fontsize=7,
        fontstyle="italic",
        color=muted,
    )

    x += 0.35
    _op(x, "=")
    x += 0.45

    mix_w = 3.4
    _name_box(x, mix_w, "mix.flac", fc="#f5c98a", fontsize=8.5)
    ax.text(
        x + mix_w / 2,
        box_y - 0.28,
        "linear sum",
        ha="center",
        va="top",
        fontsize=7,
        fontstyle="italic",
        color=muted,
    )

    fig.tight_layout()
    _savefig(fig, output_path, pad_inches=0.08)
    plt.close(fig)



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
    sep_arm_order = [a for a in _ARM_ORDER_SEP if a in set(sep["train_arm"])]
    sao_arm_order = [a for a in _ARM_ORDER_SAO if a in set(sao_df["train_arm"])]
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
                hue_order=sep_arm_order,
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
                    order=sao_arm_order,
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
