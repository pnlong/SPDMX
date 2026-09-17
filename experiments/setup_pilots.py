"""One-command setup for ICASSP pilots on a collaborator machine.

Assumes Deep Freeze is mounted (same layout as ``.env.example``). Creates
``.env`` if missing, checks SPDMX + Slakh paths, creates in-repo symlinks,
then sets up YourMT3 and stream-music-gen.

Usage (from repo root, after ``uv sync``)::

    uv run python -m experiments.setup_pilots
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from shared.config import OUTPUT_DIR, SLAKH_ROOT, SPDMX_DATASET_DIR_NAME
from shared.env import repo_root

REPO_ROOT = repo_root()
ENV_EXAMPLE = REPO_ROOT / ".env.example"
ENV_PATH = REPO_ROOT / ".env"


def _run(cmd: list[str], *, check: bool = True) -> int:
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=REPO_ROOT, check=check).returncode


def _ensure_env(*, force_example: bool = False) -> Path:
    if ENV_PATH.is_file() and not force_example:
        print(f".env already present: {ENV_PATH}")
        return ENV_PATH
    if not ENV_EXAMPLE.is_file():
        raise SystemExit(f"missing {ENV_EXAMPLE}")
    shutil.copy(ENV_EXAMPLE, ENV_PATH)
    print(f"wrote {ENV_PATH} from .env.example (edit paths if your mount differs)")
    return ENV_PATH


def _check_deepfreeze() -> dict[str, Path]:
    spdmx = Path(OUTPUT_DIR) / SPDMX_DATASET_DIR_NAME
    songs = spdmx / "songs.csv"
    stems = spdmx / "stems.csv"
    slakh = Path(SLAKH_ROOT)
    checks = {
        "SPDMX_OUTPUT_DIR": Path(OUTPUT_DIR),
        "SPDMX songs.csv": songs,
        "SPDMX stems.csv": stems,
        "Slakh root": slakh,
    }
    missing = [f"{label}: {path}" for label, path in checks.items() if not path.exists()]
    if missing:
        print("Deep Freeze path check FAILED:")
        for line in missing:
            print(f"  - {line}")
        print()
        print("Fix .env (SPDMX_OUTPUT_DIR / SPDMX_SLAKH_ROOT), ensure the mount is up,")
        print("then re-run. See experiments/COLLABORATOR_SETUP.md")
        raise SystemExit(1)
    print("Deep Freeze paths OK:")
    for label, path in checks.items():
        print(f"  {label}: {path}")
    return {"spdmx": spdmx, "slakh": slakh, "output": Path(OUTPUT_DIR)}


def _ensure_git_lfs() -> None:
    if shutil.which("git-lfs") is None:
        print(
            "warning: git-lfs not found. YourMT3 clone may fail.\n"
            "  Install: sudo apt install git-lfs && git lfs install"
        )
    else:
        _run(["git", "lfs", "install"], check=False)


def _setup_symlinks() -> None:
    _run([sys.executable, "-m", "shared.setup_symlinks"])


def _setup_yourmt3(*, with_weights: bool, skip_install: bool) -> None:
    argv = []
    if with_weights:
        argv.append("--with-weights")
    if skip_install:
        argv.append("--skip-install")
    from experiments.transcription.setup_yourmt3 import main as yourmt3_main

    yourmt3_main(argv)


def _setup_streamgen(*, with_weights: bool, with_eval: bool, skip_install: bool) -> None:
    argv = []
    if with_weights:
        argv.append("--with-weights")
    if with_eval:
        argv.append("--with-eval")
    if skip_install:
        argv.append("--skip-install")
    from experiments.streamgen.setup_streamgen import main as streamgen_main

    streamgen_main(argv)


def _write_ready(paths: dict[str, Path]) -> Path:
    md = REPO_ROOT / "experiments" / "PILOTS_READY.md"
    md.write_text(
        "\n".join(
            [
                "# ICASSP pilots ready",
                "",
                f"- SPDMX release: `{paths['spdmx']}`",
                f"- Slakh: `{paths['slakh']}`",
                f"- Output root: `{paths['output']}`",
                "",
                "## Per-experiment next steps",
                "",
                "- Transcription (YourMT3): `experiments/transcription/YOURMT3_READY.md`",
                "- StreamGen: `experiments/streamgen/STREAMGEN_READY.md`",
                "",
                "## Train stubs (metrics placeholders)",
                "",
                "```bash",
                "uv run python -m experiments.transcription.train --write-placeholder-metrics",
                "uv run python -m experiments.streamgen.train --write-placeholder-metrics",
                "```",
                "",
                "## Note on deps",
                "",
                "YourMT3 and stream-music-gen pin different `transformers` ranges.",
                "This setup installs **StreamGen last**. If a YourMT3 train breaks on",
                "`transformers`, re-run:",
                "",
                "```bash",
                "uv run python -m experiments.transcription.setup_yourmt3 --skip-clone",
                "```",
                "",
                "Or use separate venvs for the two pilots.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"wrote {md}")
    return md


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        choices=("both", "yourmt3", "streamgen", "check"),
        default="both",
        help="Which pilot(s) to set up (default: both). `check` = env + deepfreeze only.",
    )
    parser.add_argument(
        "--with-weights",
        action="store_true",
        help="Download YourMT3 checkpoints and StreamGen DAC weights (large).",
    )
    parser.add_argument(
        "--with-eval",
        action="store_true",
        help="Install StreamGen evaluation extras.",
    )
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--skip-symlinks", action="store_true")
    parser.add_argument(
        "--force-env",
        action="store_true",
        help="Overwrite .env from .env.example (destructive).",
    )
    args = parser.parse_args(argv)

    os.chdir(REPO_ROOT)
    print("=== 1/4  .env ===")
    _ensure_env(force_example=args.force_env)
    # Reload config paths after ensuring .env (shared.config already loaded once).
    # Re-read via env for the check message; values match if .env existed at import.
    print("=== 2/4  Deep Freeze ===")
    paths = _check_deepfreeze()
    if args.only == "check":
        print("Check-only mode; skipping clones.")
        return

    print("=== 3/4  symlinks + git-lfs ===")
    _ensure_git_lfs()
    if not args.skip_symlinks:
        _setup_symlinks()
    else:
        print("skipping symlinks")

    print("=== 4/4  pilot packages ===")
    if args.only in ("both", "yourmt3"):
        print("--- YourMT3 ---")
        _setup_yourmt3(with_weights=args.with_weights, skip_install=args.skip_install)
    if args.only in ("both", "streamgen"):
        print("--- StreamGen ---")
        _setup_streamgen(
            with_weights=args.with_weights,
            with_eval=args.with_eval,
            skip_install=args.skip_install,
        )

    ready = _write_ready(paths)
    print()
    print("Done. Collaborator pilot setup complete.")
    print(f"Next: open {ready}")


if __name__ == "__main__":
    main()
