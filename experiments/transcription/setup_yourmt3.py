"""One-command YourMT3 setup for the SPDMX transcription pilot.

Clones the runnable YourMT3+ pre-release (Hugging Face Spaces), installs
Python deps into the current environment, builds SPDMX/Slakh manifests if
needed, and writes a short next-steps file.

Usage (from repo root)::

    uv run python -m experiments.transcription.setup_yourmt3
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from experiments.transcription.paths import REPO_ROOT, TRANS_DIR

DEFAULT_DEST = TRANS_DIR / "YourMT3"
HF_REPO = "https://huggingface.co/spaces/mimbres/YourMT3"
PAPER_MANIFESTS = REPO_ROOT / "analysis" / "paper_data" / "transcription_manifests"

# Upstream pins an ancient CUDA torch wheel index; we skip those lines and
# assume the SPDMX env already has a working torch.
_SKIP_REQ_PREFIXES = (
    "--extra-index-url",
    "torch",
    "torchaudio",
    "yt-dlp",
    "https://github.com/coletdjnz/yt-dlp-youtube-oauth2",
    "gradio_log",  # demo-only
)


def _run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    check: bool = True,
    env: dict | None = None,
) -> int:
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=cwd, check=check, env=env).returncode


def _ensure_git_lfs() -> None:
    if shutil.which("git-lfs") is None:
        raise SystemExit(
            "git-lfs is required (even when skipping weights).\n"
            "Install it (e.g. `sudo apt install git-lfs`) then re-run."
        )
    _run(["git", "lfs", "install"], check=False)


def _clone(dest: Path, *, force: bool, with_weights: bool) -> None:
    if dest.exists():
        if force:
            print(f"removing existing {dest}")
            shutil.rmtree(dest)
        else:
            print(f"YourMT3 already at {dest} (pass --force to re-clone)")
            if with_weights:
                print("pulling LFS weights into existing clone …")
                _run(["git", "lfs", "pull"], cwd=dest, check=False)
            return
    dest.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if not with_weights:
        env["GIT_LFS_SKIP_SMUDGE"] = "1"
        print("cloning code only (skip checkpoints); re-run with --with-weights later")
    _run(["git", "clone", "--depth", "1", HF_REPO, str(dest)], env=env)
    if with_weights and dest.is_dir():
        _run(["git", "lfs", "pull"], cwd=dest, check=False)


def _filtered_requirements(src: Path, dest: Path) -> Path:
    lines_out: list[str] = []
    for raw in src.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if any(line.startswith(p) or line == p for p in _SKIP_REQ_PREFIXES):
            continue
        lines_out.append(raw)
    # Known-good pin from upstream issue discussions.
    if not any(l.startswith("transformers") for l in lines_out):
        lines_out.append("transformers==4.45.1")
    dest.write_text("\n".join(lines_out) + "\n", encoding="utf-8")
    return dest


def _install_deps(yourmt3: Path, *, skip_install: bool) -> None:
    if skip_install:
        print("skipping pip install (--skip-install)")
        return
    req = yourmt3 / "requirements.txt"
    if not req.is_file():
        raise SystemExit(f"missing {req}")
    filtered = yourmt3 / "requirements.spdmx.txt"
    _filtered_requirements(req, filtered)
    # Prefer uv if available (same env as SPDMX).
    if shutil.which("uv"):
        _run(["uv", "pip", "install", "-r", str(filtered)])
    else:
        _run([sys.executable, "-m", "pip", "install", "-r", str(filtered)])
    # Soft check torch.
    try:
        import torch  # noqa: F401

        print(f"torch OK: {torch.__version__}")
    except Exception as exc:  # pragma: no cover
        print(
            f"warning: torch import failed ({exc}). "
            "Install a GPU/CPU torch build for your machine, then re-run with --skip-clone."
        )


def _ensure_manifests(*, rebuild: bool) -> Path:
    out = PAPER_MANIFESTS
    need = rebuild or not (out / "manifest_spdmx.csv").is_file()
    if need:
        print("building transcription manifests …")
        _run([sys.executable, "-m", "experiments.transcription.prepare_manifest"])
    else:
        print(f"manifests already present under {out}")
    return out


def _write_ready(yourmt3: Path, manifests: Path) -> Path:
    src_dir = yourmt3 / "amt" / "src"
    ready = {
        "yourmt3_root": str(yourmt3),
        "yourmt3_src": str(src_dir) if src_dir.is_dir() else None,
        "manifests": str(manifests),
        "arms": ["slakh", "spdmx", "both", "spdmx_hour_matched"],
        "hf_source": HF_REPO,
        "note": (
            "GitHub mimbres/YourMT3 main is a stub; this clone is the HF Spaces "
            "pre-release that contains amt/src + checkpoints."
        ),
        "next": [
            f"cd {src_dir}" if src_dir.is_dir() else f"cd {yourmt3}",
            "export PYTHONPATH=$PWD:${PYTHONPATH:-}",
            "# Point dataset configs at the CSV manifests under analysis/paper_data/transcription_manifests/",
            "# See docs/blog/transcription.html and experiments/transcription/README.md",
        ],
    }
    path = TRANS_DIR / "YOURMT3_READY.json"
    path.write_text(json.dumps(ready, indent=2) + "\n", encoding="utf-8")
    md = TRANS_DIR / "YOURMT3_READY.md"
    md.write_text(
        "\n".join(
            [
                "# YourMT3 is set up",
                "",
                f"- Clone: `{yourmt3}`",
                f"- Source: `{src_dir if src_dir.is_dir() else yourmt3}`",
                f"- Manifests: `{manifests}`",
                "",
                "## Quick check",
                "",
                "```bash",
                f"ls {yourmt3}/amt/src",
                f"head -3 {manifests}/manifest_spdmx.csv",
                "```",
                "",
                "## Train",
                "",
                "```bash",
                "uv run python -m experiments.transcription.train --arm slakh --gpu 1",
                "uv run python -m experiments.transcription.train --arm spdmx --gpu 2",
                "```",
                "",
                "Record metrics into `analysis/paper_data/transcription_note_f1.csv` when done.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"wrote {path}")
    print(f"wrote {md}")
    return md


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--force", action="store_true", help="Re-clone YourMT3")
    parser.add_argument("--skip-clone", action="store_true")
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--rebuild-manifests", action="store_true")
    parser.add_argument(
        "--with-weights",
        action="store_true",
        help="Also download LFS checkpoints (large). Default is code-only.",
    )
    args = parser.parse_args(argv)

    os.chdir(REPO_ROOT)
    _ensure_git_lfs()
    if not args.skip_clone:
        _clone(args.dest, force=args.force, with_weights=args.with_weights)
    elif not args.dest.is_dir():
        raise SystemExit(f"--skip-clone set but missing {args.dest}")
    elif args.with_weights:
        _run(["git", "lfs", "pull"], cwd=args.dest, check=False)

    _install_deps(args.dest, skip_install=args.skip_install)
    manifests = _ensure_manifests(rebuild=args.rebuild_manifests)
    ready = _write_ready(args.dest, manifests)
    print()
    print("Done. YourMT3 setup complete.")
    print(f"Next: open {ready}")
    print(
        "Then build indexes:\n"
        "  uv run python -m experiments.transcription.manifest_to_yourmt3_indexes"
    )


if __name__ == "__main__":
    main()
