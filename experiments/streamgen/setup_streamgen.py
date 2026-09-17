"""One-command stream-music-gen setup for the SPDMX StreamGen pilot.

Clones lukewys/stream-music-gen, installs Python deps into the current
environment (keeping your existing torch), builds/reuses the SPDMX JSONL
index, and writes a short next-steps file.

Usage (from repo root)::

    uv run python -m experiments.streamgen.setup_streamgen
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.request import urlretrieve

from experiments.streamgen.paths import REPO_ROOT, SG_DIR

DEFAULT_DEST = SG_DIR / "stream-music-gen"
GH_REPO = "https://github.com/lukewys/stream-music-gen"
PAPER_INDEX = REPO_ROOT / "analysis" / "paper_data" / "streamgen_spdmx_index"
DAC_WEIGHTS_URL = (
    "https://huggingface.co/lukewys/stream_music_gen/resolve/main/"
    "250121_stemmix_dac_weights_400k_steps.pth"
)
DAC_WEIGHTS_NAME = "250121_stemmix_dac_weights_400k_steps.pth"

# Upstream pins torch 2.5.x; SPDMX already has a working torch — skip those.
_SKIP_REQ_PREFIXES = (
    "torch",
    "torchaudio",
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


def _clone(dest: Path, *, force: bool) -> None:
    if dest.exists():
        if force:
            print(f"removing existing {dest}")
            shutil.rmtree(dest)
        else:
            print(f"stream-music-gen already at {dest} (pass --force to re-clone)")
            return
    dest.parent.mkdir(parents=True, exist_ok=True)
    _run(["git", "clone", "--depth", "1", GH_REPO, str(dest)])


def _filtered_requirements(src: Path, dest: Path) -> Path:
    """Drop torch pins; keep ranges otherwise so install stays usable."""
    lines_out: list[str] = []
    for raw in src.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # Match "torch", "torch>=…", "torchaudio>=…"
        pkg = line.split("@", 1)[0].split(";", 1)[0].strip()
        name = pkg.split("[", 1)[0].split("=", 1)[0].split(">", 1)[0].split("<", 1)[0].strip()
        if any(name == p or name.startswith(f"{p}-") for p in _SKIP_REQ_PREFIXES):
            continue
        lines_out.append(raw)
    dest.write_text("\n".join(lines_out) + "\n", encoding="utf-8")
    return dest


def _relax_requires_python(upstream: Path) -> None:
    """Upstream asks for >=3.11; SPDMX often runs 3.10 — loosen for local install."""
    pyproject = upstream / "pyproject.toml"
    if pyproject.is_file():
        text = pyproject.read_text(encoding="utf-8")
        if 'requires-python = ">=3.11"' in text:
            pyproject.write_text(
                text.replace('requires-python = ">=3.11"', 'requires-python = ">=3.10"'),
                encoding="utf-8",
            )
            print("relaxed pyproject requires-python to >=3.10")
    setup_py = upstream / "setup.py"
    if setup_py.is_file():
        text = setup_py.read_text(encoding="utf-8")
        if 'python_requires=">=3.11"' in text:
            setup_py.write_text(
                text.replace('python_requires=">=3.11"', 'python_requires=">=3.10"'),
                encoding="utf-8",
            )
            print("relaxed setup.py python_requires to >=3.10")
    # Stale egg-info can keep the old Requires-Python metadata.
    for egg in upstream.glob("*.egg-info"):
        shutil.rmtree(egg)
        print(f"removed stale {egg.name}")

def _install_deps(upstream: Path, *, skip_install: bool, with_eval: bool) -> None:
    if skip_install:
        print("skipping pip install (--skip-install)")
        return
    req = upstream / "requirements.txt"
    if not req.is_file():
        raise SystemExit(f"missing {req}")
    filtered = upstream / "requirements.spdmx.txt"
    _filtered_requirements(req, filtered)
    _relax_requires_python(upstream)

    pip = ["uv", "pip"] if shutil.which("uv") else [sys.executable, "-m", "pip"]
    # Install deps first, then editable package with --no-deps so we do not
    # re-pull torch (upstream pins 2.5.x; SPDMX already has a working build).
    _run([*pip, "install", "-r", str(filtered)])
    if with_eval:
        eval_req = upstream / "requirements-eval.txt"
        if eval_req.is_file():
            _run([*pip, "install", "-r", str(eval_req)], check=False)
        else:
            print(f"warning: missing {eval_req}; skipping --with-eval")
    # Editable after relaxing requires-python (uv has no pip module in this env).
    if shutil.which("uv"):
        _run(["uv", "pip", "install", "-e", str(upstream), "--no-deps"])
    else:
        _run([sys.executable, "-m", "pip", "install", "-e", str(upstream), "--no-deps"])

    try:
        import torch  # noqa: F401

        print(f"torch OK: {torch.__version__}")
    except Exception as exc:  # pragma: no cover
        print(
            f"warning: torch import failed ({exc}). "
            "Install a GPU/CPU torch build for your machine, then re-run with --skip-clone."
        )
    # Fresh interpreter so the editable install is visible.
    rc = _run(
        [sys.executable, "-c", "import stream_music_gen; print('stream_music_gen import OK')"],
        check=False,
    )
    if rc != 0:
        print("warning: stream_music_gen import failed in a fresh process")

def _ensure_index(*, rebuild: bool) -> Path:
    index = PAPER_INDEX / "spdmx_multitrack.jsonl"
    need = rebuild or not index.is_file()
    if need:
        print("building StreamGen SPDMX index …")
        _run([sys.executable, "-m", "experiments.streamgen.prepare_spdmx_index"])
    else:
        print(f"index already present under {PAPER_INDEX}")
    # Prefer paper_data copy (always in-repo); prepare may also write to deepfreeze.
    if index.is_file():
        return PAPER_INDEX
    # Fall back: whatever prepare wrote under resolve_dev_dir.
    from experiments.streamgen.paths import resolve_dev_dir

    alt = resolve_dev_dir() / "spdmx_index"
    if (alt / "spdmx_multitrack.jsonl").is_file():
        return alt
    raise SystemExit("StreamGen index missing after prepare; see prepare_spdmx_index errors")


def _download_dac_weights(upstream: Path) -> Path | None:
    dest_dir = upstream / "pretrained_models"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / DAC_WEIGHTS_NAME
    if dest.is_file() and dest.stat().st_size > 1_000_000:
        print(f"DAC weights already at {dest}")
        return dest
    print(f"downloading DAC weights → {dest}")
    try:
        urlretrieve(DAC_WEIGHTS_URL, dest)
    except Exception as exc:
        print(f"warning: DAC download failed ({exc})")
        print(f"  Manual: place {DAC_WEIGHTS_NAME} in {dest_dir}")
        return None
    print(f"wrote {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def _write_ready(upstream: Path, index_dir: Path, dac: Path | None) -> Path:
    ready = {
        "upstream_root": str(upstream),
        "index_dir": str(index_dir),
        "index": str(index_dir / "spdmx_multitrack.jsonl"),
        "hour_matched": str(index_dir / "spdmx_hour_matched.jsonl"),
        "dac_weights": str(dac) if dac else None,
        "github": GH_REPO,
        "pilot": {
            "script": "scripts/train_dec_online.py",
            "config": "configs/online/decoder_online_future_visibility_0.yml",
            "future_visibility": 0,
        },
        "note": (
            "Upstream requires Python >=3.11 and pins torch 2.5.x; this setup "
            "keeps your existing torch and installs the package with --no-deps."
        ),
        "next": [
            f"cd {upstream}",
            "# Wire SPDMX JSONL adapter (see ADAPTER.md in the index dir)",
            "# extract_causal_dac_32k → extract_rms → dump_audio_mixdown",
            "python scripts/train_dec_online.py "
            "--args.load configs/online/decoder_online_future_visibility_0.yml "
            "--save_dir logs/dec_online_future_visibility_0",
        ],
    }
    path = SG_DIR / "STREAMGEN_READY.json"
    path.write_text(json.dumps(ready, indent=2) + "\n", encoding="utf-8")
    md = SG_DIR / "STREAMGEN_READY.md"
    dac_line = f"- DAC weights: `{dac}`" if dac else (
        "- DAC weights: not downloaded (re-run with `--with-weights`)"
    )
    md.write_text(
        "\n".join(
            [
                "# StreamGen is set up",
                "",
                f"- Clone: `{upstream}`",
                f"- SPDMX index: `{index_dir}`",
                dac_line,
                "",
                "## Quick check",
                "",
                "```bash",
                f"ls {upstream}/stream_music_gen",
                f"head -1 {index_dir}/spdmx_multitrack.jsonl",
                "```",
                "",
                "## Next (upstream)",
                "",
                "1. Follow `{}/ADAPTER.md` to add an SPDMX dataset class.".format(index_dir),
                "2. Dump DAC / RMS / mixdown features (needs the DAC weights).",
                "3. Train the pilot:",
                "",
                "```bash",
                f"cd {upstream}",
                "python scripts/train_dec_online.py \\",
                "  --args.load configs/online/decoder_online_future_visibility_0.yml \\",
                "  --save_dir logs/dec_online_future_visibility_0",
                "```",
                "",
                "Record metrics into `analysis/paper_data/streamgen_metrics.csv` when done.",
                "",
                "## Notes",
                "",
                "- Setup skips upstream `torch`/`torchaudio` pins so your SPDMX torch stays.",
                "- If you also installed YourMT3, `transformers` versions may conflict; "
                "use a separate venv if training both in parallel.",
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
    parser.add_argument("--force", action="store_true", help="Re-clone stream-music-gen")
    parser.add_argument("--skip-clone", action="store_true")
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument(
        "--with-eval",
        action="store_true",
        help="Also install optional evaluation extras (COCOLA / FAD / etc.).",
    )
    parser.add_argument(
        "--with-weights",
        action="store_true",
        help="Download causal DAC checkpoint into pretrained_models/ (large).",
    )
    args = parser.parse_args(argv)

    os.chdir(REPO_ROOT)
    if not args.skip_clone:
        _clone(args.dest, force=args.force)
    elif not args.dest.is_dir():
        raise SystemExit(f"--skip-clone set but missing {args.dest}")

    _install_deps(args.dest, skip_install=args.skip_install, with_eval=args.with_eval)
    index_dir = _ensure_index(rebuild=args.rebuild_index)
    dac = _download_dac_weights(args.dest) if args.with_weights else None
    ready = _write_ready(args.dest, index_dir, dac)
    print()
    print("Done. StreamGen setup complete.")
    print(f"Next: open {ready}")


if __name__ == "__main__":
    main()
