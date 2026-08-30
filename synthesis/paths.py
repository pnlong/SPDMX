"""Output path helpers."""

from __future__ import annotations

from pathlib import Path

from shared.config import (
    ABLATIONS_DIR_NAME,
    ANALYSIS_DIR_NAME,
    DEV_DIR_NAME,
    INSTRUMENTS_DIR_NAME,
    MID_CORRECTED_DIR_NAME,
    SONG_LENGTHS_DIR_NAME,
    SPDMX_AUDIO_DIR_NAME,
    SPDMX_DATASET_DIR_NAME,
    SPDMX_MID_DIR_NAME,
    SPDMX_RAW_DIR_NAME,
    STEMS_DIR_NAME,
    STEMS_REALIFY_DIR_NAME,
    TRACK_NAMES_DIR_NAME,
)

# {OUTPUT_DIR}/dev/final/ — production data.csv / stems.csv / stem_recipe.csv
PRODUCTION_TABLES_DIR_NAME = "final"
# One row per song: dense MIDI path + n_tracks (avoids per-song NFS stat at render start)
MIDI_INDEX_FILE_NAME = "midi_index.csv"
PASS_TRACK_COLUMNS = {
    "fluidsynth": "n_fluidsynth",
    "ddsp_piano": "n_ddsp_piano",
    "midi_ddsp": "n_midi_ddsp",
}


def condition_name(render_mode: str, realify: bool = False) -> str:
    return f"{render_mode}_realify" if realify else render_mode


def dev_root(output_dir: str) -> str:
    return f"{output_dir}/{DEV_DIR_NAME}"


def ablations_root(output_dir: str) -> str:
    return f"{dev_root(output_dir)}/{ABLATIONS_DIR_NAME}"


def ablation_dir(output_dir: str, condition: str) -> str:
    return f"{ablations_root(output_dir)}/{condition}"


def ablation_raw_dir(output_dir: str, render_mode: str) -> str:
    return ablation_dir(output_dir, render_mode)


def ablation_realify_dir(output_dir: str, render_mode: str) -> str:
    return ablation_dir(output_dir, condition_name(render_mode, realify=True))


def full_stems_dir(output_dir: str) -> str:
    return f"{dev_root(output_dir)}/{STEMS_DIR_NAME}"


def full_stems_realify_dir(output_dir: str) -> str:
    return f"{dev_root(output_dir)}/{STEMS_REALIFY_DIR_NAME}"


def analysis_root(output_dir: str) -> str:
    return f"{dev_root(output_dir)}/{ANALYSIS_DIR_NAME}"


def mid_corrected_dir(output_dir: str) -> str:
    """Dense corrected MIDI tree (same as released ``SPDMX/mid/``)."""
    return spdmx_mid_dir(output_dir)


def legacy_mid_corrected_dir(output_dir: str) -> str:
    """Previous location: ``{OUTPUT_DIR}/dev/mid_corrected/``."""
    return f"{dev_root(output_dir)}/{MID_CORRECTED_DIR_NAME}"


def song_lengths_dir(output_dir: str) -> str:
    return f"{analysis_root(output_dir)}/{SONG_LENGTHS_DIR_NAME}"


def instruments_dir(output_dir: str) -> str:
    return f"{analysis_root(output_dir)}/{INSTRUMENTS_DIR_NAME}"


def track_names_dir(output_dir: str) -> str:
    return f"{analysis_root(output_dir)}/{TRACK_NAMES_DIR_NAME}"


def production_tables_dir(output_dir: str) -> str:
    """Bookkeeping CSVs for hybrid ``synthesis.final --full`` (not the released tree)."""
    return f"{dev_root(output_dir)}/{PRODUCTION_TABLES_DIR_NAME}"


def spdmx_dataset_dir(output_dir: str) -> str:
    return f"{output_dir}/{SPDMX_DATASET_DIR_NAME}"


def spdmx_audio_dir(output_dir: str) -> str:
    """Released / mixable stem tree: ``{OUTPUT}/SPDMX/audio/``."""
    return f"{spdmx_dataset_dir(output_dir)}/{SPDMX_AUDIO_DIR_NAME}"


def spdmx_raw_dir(output_dir: str) -> str:
    """Pre-mix hybrid stem tree: ``{OUTPUT}/SPDMX/raw/``."""
    return f"{spdmx_dataset_dir(output_dir)}/{SPDMX_RAW_DIR_NAME}"


def spdmx_mid_dir(output_dir: str) -> str:
    return f"{spdmx_dataset_dir(output_dir)}/{SPDMX_MID_DIR_NAME}"


def raw_path_to_audio(path: str) -> str:
    """Map ``…/raw/…`` or ``./raw/…`` to the sibling released ``audio/`` tree."""
    text = str(path)
    rel_src = f"./{SPDMX_RAW_DIR_NAME}/"
    rel_dst = f"./{SPDMX_AUDIO_DIR_NAME}/"
    if text.startswith(rel_src):
        return rel_dst + text[len(rel_src):]
    abs_src = f"/{SPDMX_RAW_DIR_NAME}/"
    abs_dst = f"/{SPDMX_AUDIO_DIR_NAME}/"
    if abs_src in text:
        return text.replace(abs_src, abs_dst, 1)
    raise ValueError(
        f"Cannot map to {SPDMX_AUDIO_DIR_NAME}: path has no "
        f"{SPDMX_RAW_DIR_NAME!r} segment: {path}"
    )


def remap_path_prefix(value: str, source_dir: Path, output_dir: Path) -> str:
    """Rewrite an absolute path under ``source_dir`` to the mirrored path under ``output_dir``."""
    source_prefix = str(source_dir)
    if value.startswith(source_prefix):
        return str(output_dir) + value[len(source_prefix):]
    return value


def resolve_output_song_dir(song_dir: Path, source_dir: Path, output_dir: Path) -> Path:
    """Map a song directory under ``source_dir`` to the mirrored path under ``output_dir``."""
    if output_dir == source_dir:
        return song_dir
    remapped = remap_path_prefix(str(song_dir), source_dir, output_dir)
    if remapped == str(song_dir):
        raise ValueError(f"Song path {song_dir} is not under source dir {source_dir}")
    return Path(remapped)
