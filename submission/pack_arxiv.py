#!/usr/bin/env python3
"""Build an arXiv source zip from the dataset preprint.

Copies ``preprint.tex`` (not ICASSP ``main.tex``), style/bst files, and
referenced figures; writes a BibTeX file with only cited entries; zips a
flat upload tree.

Usage::

    uv run python -m submission.pack_arxiv
    uv run python -m submission.pack_arxiv --out submission/arxiv_upload.zip
"""

from __future__ import annotations

import argparse
import re
import shutil
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_TEX = HERE / "preprint.tex"
DEFAULT_BIB = HERE / "main.bib"
DEFAULT_OUT = HERE / "arxiv_upload.zip"
DEFAULT_STAGE = HERE / "arxiv_build"

_CITE_RE = re.compile(r"\\cite[a-zA-Z]*\*?\s*\{([^}]+)\}")
_GRAPHICS_RE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
_BIBLIOGRAPHY_RE = re.compile(r"\\bibliography\{([^}]+)\}")
_ENTRY_START_RE = re.compile(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", re.MULTILINE)


def cited_keys(tex: str) -> set[str]:
    keys: set[str] = set()
    for match in _CITE_RE.finditer(tex):
        for key in match.group(1).split(","):
            key = key.strip()
            if key:
                keys.add(key)
    return keys


def graphics_paths(tex: str) -> list[str]:
    return [m.group(1).strip() for m in _GRAPHICS_RE.finditer(tex)]


def bibliography_basenames(tex: str) -> list[str]:
    names: list[str] = []
    for match in _BIBLIOGRAPHY_RE.finditer(tex):
        for part in match.group(1).split(","):
            part = part.strip()
            if part:
                names.append(part)
    return names or ["main"]


def _consume_braced(text: str, open_idx: int) -> int:
    """Return index after the closing ``}`` matching ``text[open_idx] == '{'``.

    Counts ``{`` / ``}`` only. BibTeX accent commands like ``{\\"o}`` must not be
    treated as quoted strings (a naive ``"`` skipper breaks on ``\\"``).
    """
    if open_idx >= len(text) or text[open_idx] != "{":
        raise ValueError("expected '{' at entry body start")
    depth = 0
    i = open_idx
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    raise ValueError("unbalanced braces in BibTeX entry")


def parse_bib_entries(bib_text: str) -> list[tuple[str, str]]:
    """Return ``[(key, full_entry_text), ...]`` in file order."""
    entries: list[tuple[str, str]] = []
    for match in _ENTRY_START_RE.finditer(bib_text):
        key = match.group(2)
        brace_open = bib_text.find("{", match.start())
        end = _consume_braced(bib_text, brace_open)
        entry = bib_text[match.start() : end].strip()
        if not entry.endswith("\n"):
            entry += "\n"
        entries.append((key, entry))
    return entries


def scrub_bib(bib_text: str, keep: set[str]) -> tuple[str, list[str], list[str]]:
    """Keep only cited entries. Returns scrubbed text, kept keys, dropped keys."""
    entries = parse_bib_entries(bib_text)
    kept: list[str] = []
    dropped: list[str] = []
    chunks: list[str] = []
    for key, entry in entries:
        if key in keep:
            kept.append(key)
            chunks.append(entry.rstrip() + "\n")
        else:
            dropped.append(key)
    missing = sorted(keep - set(kept))
    if missing:
        raise SystemExit(
            "cited keys missing from BibTeX: " + ", ".join(missing)
        )
    body = "\n".join(chunks)
    if body and not body.endswith("\n"):
        body += "\n"
    return body, kept, dropped


def pack_arxiv(
    *,
    tex_path: Path,
    bib_path: Path,
    out_zip: Path,
    stage_dir: Path,
    style_files: list[str] | None = None,
) -> dict:
    tex = tex_path.read_text(encoding="utf-8")
    bib = bib_path.read_text(encoding="utf-8")
    keys = cited_keys(tex)
    scrubbed, kept, dropped = scrub_bib(bib, keys)
    figs = graphics_paths(tex)
    bib_names = bibliography_basenames(tex)

    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True)

    # Primary manuscript only (no ICASSP main.tex).
    shutil.copy2(tex_path, stage_dir / tex_path.name)

    # BibTeX: keep the basename(s) referenced by \\bibliography{...}.
    for name in bib_names:
        dest = stage_dir / f"{name}.bib"
        dest.write_text(scrubbed, encoding="utf-8")

    styles = style_files or ["spconf.sty", "IEEEbib.bst"]
    for name in styles:
        src = HERE / name
        if not src.is_file():
            raise SystemExit(f"missing style file: {src}")
        shutil.copy2(src, stage_dir / name)

    for rel in figs:
        src = (tex_path.parent / rel).resolve()
        if not src.is_file():
            # Try with common image extensions if omitted.
            found = None
            for ext in (".pdf", ".png", ".jpg", ".jpeg", ".eps"):
                cand = Path(str(src) + ext)
                if cand.is_file():
                    found = cand
                    rel = rel + ext
                    break
            if found is None:
                raise SystemExit(f"missing figure: {rel} (looked for {src})")
            src = found
        dest = stage_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)

    # Refuse to ship ICASSP main.tex even if present next to preprint.
    banned = {"main.tex"}
    for path in stage_dir.rglob("*"):
        if path.is_file() and path.name in banned:
            path.unlink()

    out_zip.parent.mkdir(parents=True, exist_ok=True)
    if out_zip.exists():
        out_zip.unlink()
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(stage_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(stage_dir).as_posix())

    return {
        "tex": tex_path.name,
        "zip": str(out_zip),
        "stage": str(stage_dir),
        "cited": sorted(keys),
        "kept": kept,
        "dropped": dropped,
        "figures": figs,
        "members": sorted(
            p.relative_to(stage_dir).as_posix()
            for p in stage_dir.rglob("*")
            if p.is_file()
        ),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tex",
        type=Path,
        default=DEFAULT_TEX,
        help=f"preprint source (default: {DEFAULT_TEX})",
    )
    parser.add_argument(
        "--bib",
        type=Path,
        default=DEFAULT_BIB,
        help=f"BibTeX database (default: {DEFAULT_BIB})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"output zip (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--stage",
        type=Path,
        default=DEFAULT_STAGE,
        help=f"staging directory (default: {DEFAULT_STAGE})",
    )
    parser.add_argument(
        "--keep-stage",
        action="store_true",
        help="keep staging directory after zipping (default: keep)",
    )
    args = parser.parse_args(argv)

    info = pack_arxiv(
        tex_path=args.tex.resolve(),
        bib_path=args.bib.resolve(),
        out_zip=args.out.resolve(),
        stage_dir=args.stage.resolve(),
    )
    print(f"tex:      {info['tex']}")
    print(f"cited:    {len(info['cited'])}")
    print(f"kept:     {len(info['kept'])}")
    if info["dropped"]:
        print(f"dropped:  {', '.join(info['dropped'])}")
    print(f"figures:  {len(info['figures'])}")
    print(f"members:  {', '.join(info['members'])}")
    print(f"wrote:    {info['zip']}")
    print(f"stage:    {info['stage']}")


if __name__ == "__main__":
    main()
