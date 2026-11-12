"""Convert paper social-blast figures from PDF to PNG.

Usage:
    uv run python -m submission.social.export_fig_pngs
"""

from __future__ import annotations

from pathlib import Path

FIGS = Path(__file__).resolve().parents[1] / "figs"
OUT = Path(__file__).resolve().parent

NAMES = (
    "stems_per_song",
    "gm_program_counts_compare",
    "ablation_listening",
)

# Tweet-facing aliases (gm figure renamed for the blast draft)
ALIASES = {
    "gm_program_counts_compare": "gm_programs",
}


def _convert_with_pdftoppm(pdf: Path, dest_stem: Path) -> bool:
    import subprocess

    # pdftoppm writes dest_stem-1.png when using -singlefile it writes dest_stem.png
    result = subprocess.run(
        [
            "pdftoppm",
            "-png",
            "-r",
            "200",
            "-singlefile",
            str(pdf),
            str(dest_stem),
        ],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and dest_stem.with_suffix(".png").exists()


def _convert_with_pymupdf(pdf: Path, png: Path) -> None:
    import fitz  # PyMuPDF

    doc = fitz.open(pdf)
    page = doc[0]
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
    pix.save(str(png))
    doc.close()


def _convert_with_pdf2image(pdf: Path, png: Path) -> None:
    from pdf2image import convert_from_path

    images = convert_from_path(str(pdf), dpi=200)
    images[0].save(png, "PNG")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name in NAMES:
        pdf = FIGS / f"{name}.pdf"
        if not pdf.exists():
            raise FileNotFoundError(pdf)
        alias = ALIASES.get(name, name)
        png = OUT / f"{alias}.png"
        dest_stem = OUT / alias

        if _convert_with_pdftoppm(pdf, dest_stem):
            print(f"Wrote {png} (pdftoppm)")
            continue

        try:
            _convert_with_pymupdf(pdf, png)
            print(f"Wrote {png} (pymupdf)")
            continue
        except Exception as exc:  # noqa: BLE001
            last = exc

        try:
            _convert_with_pdf2image(pdf, png)
            print(f"Wrote {png} (pdf2image)")
            continue
        except Exception as exc:  # noqa: BLE001
            last = exc

        raise RuntimeError(
            f"Could not convert {pdf}. Install poppler (pdftoppm), PyMuPDF, or pdf2image."
        ) from last


if __name__ == "__main__":
    main()
