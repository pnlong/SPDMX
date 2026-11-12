"""Render a wide, phone-readable datasets comparison table for the Twitter blast.

Usage:
    uv run python -m submission.social.render_datasets_table
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUT = Path(__file__).resolve().parent / "datasets_table.png"

# Same rows/numbers as submission/main.tex tab:datasets; columns fully spelled out.
HEADERS = [
    "Dataset",
    "Songs",
    "Mix hours",
    "Type",
    "Open download",
    "Open synth\n(no proprietary VSTs)",
]

ROWS = [
    ("MUSDB18", "150", "10", "Recorded", "No", "—"),
    ("MedleyDB + MedleyDB 2.0", "196", "13.8", "Recorded", "No", "—"),
    ("MoisesDB", "240", "14.4", "Recorded", "No", "—"),
    ("Slakh2100*", "2,100", "145", "Synthetic", "Yes", "No"),
    ("CocoChorales", "240,000", "1,411", "Synthetic", "Yes", "Yes"),
    ("EnsembleSet", "80", "6.15", "Synthetic", "Yes", "No"),
    ("SPDMX (All)", "253,982", "5,469", "Synthetic", "Yes", "Yes"),
    ("SPDMX (Multitrack)", "88,403", "2,734", "Synthetic", "Yes", "Yes"),
]

FOOTNOTE = (
    "*Slakh2100: Suspect CC BY audio, but proprietary VST renderers.  "
    "SPDMX includes single-stem songs; multitrack = ≥2 stems.  "
    "MedleyDB = MedleyDB ∪ MedleyDB 2.0."
)


def main() -> None:
    n_rows = len(ROWS)
    n_cols = len(HEADERS)

    # Wide landscape for Twitter readability
    fig_w, fig_h = 16.0, 7.2
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    # Column widths (sum = 1)
    col_w = [0.22, 0.11, 0.11, 0.14, 0.16, 0.26]
    assert abs(sum(col_w) - 1.0) < 1e-9

    top = 0.92
    header_h = 0.12
    row_h = 0.085
    left = 0.02
    usable_w = 0.96

    def x_edges() -> list[float]:
        xs = [left]
        for w in col_w:
            xs.append(xs[-1] + w * usable_w)
        return xs

    xs = x_edges()

    # Title
    ax.text(
        0.5,
        0.975,
        "Multitrack resources compared to SPDMX",
        ha="center",
        va="top",
        fontsize=18,
        fontweight="bold",
        color="#111111",
        transform=ax.transAxes,
    )

    # Header background
    header_bottom = top - header_h
    ax.add_patch(
        Rectangle(
            (left, header_bottom),
            usable_w,
            header_h,
            facecolor="#1a1a1a",
            edgecolor="#1a1a1a",
            linewidth=0.5,
            transform=ax.transAxes,
            clip_on=False,
        )
    )

    for i, header in enumerate(HEADERS):
        cx = (xs[i] + xs[i + 1]) / 2
        ax.text(
            cx,
            header_bottom + header_h / 2,
            header,
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color="#ffffff",
            transform=ax.transAxes,
            linespacing=1.15,
        )

    # Body rows
    for r, row in enumerate(ROWS):
        y_top = header_bottom - r * row_h
        y_bot = y_top - row_h
        is_spdmx = row[0].startswith("SPDMX")
        # Separator before SPDMX block
        if r == 6:
            ax.plot(
                [left, left + usable_w],
                [y_top, y_top],
                color="#333333",
                linewidth=1.6,
                transform=ax.transAxes,
                clip_on=False,
            )

        if is_spdmx:
            facecolor = "#e8f5e9"
        elif r % 2 == 0:
            facecolor = "#f7f7f7"
        else:
            facecolor = "#ffffff"

        ax.add_patch(
            Rectangle(
                (left, y_bot),
                usable_w,
                row_h,
                facecolor=facecolor,
                edgecolor="#dddddd",
                linewidth=0.6,
                transform=ax.transAxes,
                clip_on=False,
            )
        )

        weight = "bold" if is_spdmx else "normal"
        color = "#0d47a1" if is_spdmx else "#222222"
        for c, cell in enumerate(row):
            cx = (xs[c] + xs[c + 1]) / 2
            ha = "left" if c == 0 else "center"
            x = xs[c] + 0.01 * usable_w if c == 0 else cx
            ax.text(
                x,
                y_bot + row_h / 2,
                cell,
                ha=ha,
                va="center",
                fontsize=12 if not is_spdmx else 12.5,
                fontweight=weight,
                color=color,
                transform=ax.transAxes,
            )

    # Vertical grid lines
    for x in xs:
        ax.plot(
            [x, x],
            [header_bottom - n_rows * row_h, top],
            color="#cccccc",
            linewidth=0.5,
            transform=ax.transAxes,
            clip_on=False,
        )

    ax.text(
        0.5,
        0.035,
        FOOTNOTE,
        ha="center",
        va="center",
        fontsize=9,
        color="#555555",
        transform=ax.transAxes,
    )

    fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
