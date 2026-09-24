"""Tests for in-repo dev artifact symlinks."""

from pathlib import Path

from shared.repo_symlinks import (
    REPO_ABLATIONS_SYMLINK,
    REPO_ANALYSIS_SYMLINK,
    REPO_PATCH_SWEEP_OUTPUT_SYMLINK,
    REPO_SOUNDFONTS_SYMLINK,
    link_ablations_in_repo,
    link_analysis_in_repo,
    link_patch_sweep_output_in_repo,
    link_soundfonts_in_repo,
    setup_dev_symlinks,
)
from experiments.paths import patch_sweep_output_root
from synthesis.paths import ablations_root, analysis_root


def test_link_analysis_in_repo(tmp_path: Path, monkeypatch):
    spdmx_root = str(tmp_path / "SPDMX")
    analysis_dir = Path(analysis_root(spdmx_root))
    analysis_dir.mkdir(parents=True)
    (analysis_dir / "song_lengths").mkdir()

    symlink = tmp_path / "repo" / "analysis" / "output"
    symlink.parent.mkdir(parents=True)
    monkeypatch.setattr("shared.repo_symlinks.REPO_ANALYSIS_SYMLINK", symlink)
    monkeypatch.setattr("shared.repo_symlinks.LEGACY_ANALYSIS_SYMLINKS", ())

    link, target = link_analysis_in_repo(spdmx_root)
    assert link == symlink
    assert symlink.is_symlink()
    assert symlink.resolve() == analysis_dir.resolve()


def test_link_ablations_in_repo(tmp_path: Path, monkeypatch):
    spdmx_root = str(tmp_path / "SPDMX")
    ablations_dir = Path(ablations_root(spdmx_root))
    ablations_dir.mkdir(parents=True)
    (ablations_dir / "basic").mkdir()

    symlink = tmp_path / "repo" / "synthesis" / "ablations_output"
    symlink.parent.mkdir(parents=True)
    monkeypatch.setattr("shared.repo_symlinks.REPO_ABLATIONS_SYMLINK", symlink)

    link, target = link_ablations_in_repo(spdmx_root)
    assert link == symlink
    assert symlink.is_symlink()
    assert symlink.resolve() == ablations_dir.resolve()
    assert (symlink / "basic").is_dir()


def test_setup_dev_symlinks(tmp_path: Path, monkeypatch):
    spdmx_root = str(tmp_path / "SPDMX")
    Path(analysis_root(spdmx_root)).mkdir(parents=True)
    Path(ablations_root(spdmx_root)).mkdir(parents=True)
    Path(patch_sweep_output_root(spdmx_root)).mkdir(parents=True)

    analysis_link = tmp_path / "repo" / "analysis" / "output"
    ablations_link = tmp_path / "repo" / "synthesis" / "ablations_output"
    patch_sweep_link = tmp_path / "repo" / "experiments" / "patch_sweep" / "output"
    soundfonts_link = tmp_path / "repo" / "soundfonts"
    analysis_link.parent.mkdir(parents=True)
    ablations_link.parent.mkdir(parents=True)
    patch_sweep_link.parent.mkdir(parents=True)
    soundfont_dir = tmp_path / "soundfonts_data"
    soundfont_dir.mkdir()
    (soundfont_dir / "SGM-V2.01.sf2").write_bytes(b"sf2")
    monkeypatch.setattr("shared.repo_symlinks.REPO_ANALYSIS_SYMLINK", analysis_link)
    monkeypatch.setattr("shared.repo_symlinks.REPO_ABLATIONS_SYMLINK", ablations_link)
    monkeypatch.setattr(
        "shared.repo_symlinks.REPO_PATCH_SWEEP_OUTPUT_SYMLINK",
        patch_sweep_link,
    )
    monkeypatch.setattr(
        "shared.repo_symlinks.REPO_SOUNDFONTS_SYMLINK",
        soundfonts_link,
    )
    monkeypatch.setattr("shared.repo_symlinks.LEGACY_ANALYSIS_SYMLINKS", ())

    links = setup_dev_symlinks(spdmx_root, soundfont_dir=str(soundfont_dir))
    assert len(links) == 4
    assert analysis_link.is_symlink()
    assert ablations_link.is_symlink()
    assert patch_sweep_link.is_symlink()
    assert soundfonts_link.is_symlink()


def test_link_soundfonts_in_repo(tmp_path: Path, monkeypatch):
    soundfont_dir = tmp_path / "soundfonts_data"
    soundfont_dir.mkdir()
    (soundfont_dir / "SGM-V2.01.sf2").write_bytes(b"sf2")

    symlink = tmp_path / "repo" / "soundfonts"
    symlink.parent.mkdir(parents=True)
    monkeypatch.setattr(
        "shared.repo_symlinks.REPO_SOUNDFONTS_SYMLINK",
        symlink,
    )

    link, target = link_soundfonts_in_repo(str(soundfont_dir))
    assert link == symlink
    assert symlink.is_symlink()
    assert symlink.resolve() == soundfont_dir.resolve()
    assert (symlink / "SGM-V2.01.sf2").is_file()


def test_link_patch_sweep_output_in_repo(tmp_path: Path, monkeypatch):
    spdmx_root = str(tmp_path / "SPDMX")
    output_dir = Path(patch_sweep_output_root(spdmx_root))
    output_dir.mkdir(parents=True)
    (output_dir / "manifest.csv").write_text("variant_id\n")

    symlink = tmp_path / "repo" / "experiments" / "patch_sweep" / "output"
    symlink.parent.mkdir(parents=True)
    monkeypatch.setattr(
        "shared.repo_symlinks.REPO_PATCH_SWEEP_OUTPUT_SYMLINK",
        symlink,
    )

    link, target = link_patch_sweep_output_in_repo(spdmx_root)
    assert link == symlink
    assert symlink.is_symlink()
    assert symlink.resolve() == output_dir.resolve()
