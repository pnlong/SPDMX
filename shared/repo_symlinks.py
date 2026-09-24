"""In-repo symlinks to deepfreeze dev artifact directories."""

from __future__ import annotations

from pathlib import Path

from shared.config import OUTPUT_DIR, SOUNDFONT_DIR
from experiments.paths import patch_sweep_output_root
from synthesis.paths import ablations_root, analysis_root

REPO_ROOT = Path(__file__).resolve().parent.parent
REPO_ANALYSIS_SYMLINK = REPO_ROOT / "analysis" / "output"
REPO_ABLATIONS_SYMLINK = REPO_ROOT / "synthesis" / "ablations_output"
REPO_PATCH_SWEEP_OUTPUT_SYMLINK = (
    REPO_ROOT / "experiments" / "patch_sweep" / "output"
)
REPO_SOUNDFONTS_SYMLINK = REPO_ROOT / "soundfonts"

LEGACY_ANALYSIS_SYMLINKS = (
    REPO_ROOT / "analysis" / "song_lengths",
    REPO_ROOT / "analysis" / "song_length_report.json",
)


def _link_repo_dir(link: Path, target: Path) -> tuple[Path, Path]:
    target = target.resolve()
    target.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.is_file():
        link.unlink()
    elif link.is_dir():
        raise RuntimeError(f"Refusing to replace real directory {link}")
    link.symlink_to(target, target_is_directory=True)
    return link, target


def link_analysis_in_repo(output_dir: str = OUTPUT_DIR) -> tuple[Path, Path]:
    """Symlink {OUTPUT_DIR}/dev/analysis into analysis/output in this repo."""
    for legacy in LEGACY_ANALYSIS_SYMLINKS:
        if legacy.is_symlink() or legacy.is_file():
            legacy.unlink()
        elif legacy.is_dir():
            raise RuntimeError(f"Refusing to replace real directory {legacy}")
    return _link_repo_dir(REPO_ANALYSIS_SYMLINK, Path(analysis_root(output_dir)))


def link_ablations_in_repo(output_dir: str = OUTPUT_DIR) -> tuple[Path, Path]:
    """Symlink {OUTPUT_DIR}/dev/ablations into synthesis/ablations_output in this repo."""
    return _link_repo_dir(REPO_ABLATIONS_SYMLINK, Path(ablations_root(output_dir)))


def link_patch_sweep_output_in_repo(output_dir: str = OUTPUT_DIR) -> tuple[Path, Path]:
    """Symlink patch sweep output into experiments/patch_sweep/output in this repo."""
    REPO_PATCH_SWEEP_OUTPUT_SYMLINK.parent.mkdir(parents=True, exist_ok=True)
    return _link_repo_dir(
        REPO_PATCH_SWEEP_OUTPUT_SYMLINK,
        Path(patch_sweep_output_root(output_dir)),
    )


def link_soundfonts_in_repo(
    soundfont_dir: str = SOUNDFONT_DIR,
) -> tuple[Path, Path]:
    """Symlink the local soundfont library into soundfonts/ at repo root."""
    target = Path(soundfont_dir)
    if not target.is_dir():
        raise FileNotFoundError(f"Soundfont directory not found: {target}")
    return _link_repo_dir(REPO_SOUNDFONTS_SYMLINK, target)


def setup_dev_symlinks(
    output_dir: str = OUTPUT_DIR,
    *,
    soundfont_dir: str = SOUNDFONT_DIR,
) -> list[tuple[Path, Path]]:
    """Create all in-repo dev artifact symlinks."""
    return [
        link_analysis_in_repo(output_dir),
        link_ablations_in_repo(output_dir),
        link_patch_sweep_output_in_repo(output_dir),
        link_soundfonts_in_repo(soundfont_dir),
    ]
