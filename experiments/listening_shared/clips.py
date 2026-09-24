"""Shared fixed-length clip helpers for listening / sweep tooling.

Recovered from the former ``experiments.preset_sweep.diverse_stems`` module after
SA3 realify / preset_sweep removal.
"""

from __future__ import annotations

import os
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import yaml

from shared.config import DATA_DIR_NAME, SAMPLE_RATE, STEMS_FILE_NAME
from synthesis.audio import stem_is_valid

DEFAULT_CLIP_SECONDS = 10.0
DEFAULT_MIN_RMS = 0.01

CLIPS_SOURCE_MARKER = "clips.source"

PRESET_SWEEP_REMOVED = (
    "experiments.preset_sweep was removed with SA3 realify; "
    "preset sweep listening / verification is no longer available."
)


def stem_rms(path: Path) -> float:
    """Root-mean-square amplitude of a stem file."""
    if not path.is_file():
        return 0.0
    try:
        audio, _ = sf.read(str(path), dtype="float32", always_2d=True)
    except (RuntimeError, OSError, ValueError):
        return 0.0
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio))))


def is_silent(path: Path, *, min_rms: float = DEFAULT_MIN_RMS) -> bool:
    return stem_rms(path) < min_rms


def clip_stem_waveform(
    path: Path,
    *,
    clip_seconds: float,
    start_seconds: float = 0.0,
) -> np.ndarray:
    start_frame = int(start_seconds * SAMPLE_RATE)
    max_frames = int(clip_seconds * SAMPLE_RATE)
    audio, _ = sf.read(
        str(path),
        start=start_frame,
        frames=max_frames,
        dtype="float32",
        always_2d=True,
    )
    if audio.ndim == 1:
        audio = audio[:, np.newaxis]
    return np.asarray(audio.T, dtype=np.float32)


def clip_rms(path: Path, *, clip_seconds: float, start_seconds: float = 0.0) -> float:
    audio = clip_stem_waveform(
        path,
        clip_seconds=clip_seconds,
        start_seconds=start_seconds,
    )
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio))))


def waveform_active_fraction(
    waveform: np.ndarray,
    *,
    sample_rate: int = SAMPLE_RATE,
    hop_seconds: float = 0.25,
    min_rms: float = DEFAULT_MIN_RMS,
) -> float:
    """Fraction of short hops inside ``waveform`` whose RMS is >= ``min_rms``."""
    if waveform.size == 0:
        return 0.0
    if waveform.ndim == 2:
        mono = np.mean(np.asarray(waveform, dtype=np.float32), axis=0)
    else:
        mono = np.asarray(waveform, dtype=np.float32).reshape(-1)

    hop = max(1, int(hop_seconds * sample_rate))
    n = int(mono.shape[0])
    if n <= 0:
        return 0.0
    if n <= hop:
        rms = float(np.sqrt(np.mean(np.square(mono))))
        return 1.0 if rms >= min_rms else 0.0

    starts = list(range(0, n - hop + 1, hop))
    if starts[-1] + hop < n:
        starts.append(n - hop)

    active = 0
    for start in starts:
        chunk = mono[start : start + hop]
        rms = float(np.sqrt(np.mean(np.square(chunk))))
        if rms >= min_rms:
            active += 1
    return active / len(starts)


def find_audible_clip_start(
    path: Path,
    *,
    clip_seconds: float,
    min_rms: float = DEFAULT_MIN_RMS,
    hop_seconds: float = 1.0,
    min_start_seconds: float = 0.0,
    min_active_fraction: float = 0.0,
    activity_hop_seconds: float = 0.25,
    prefer_densest: bool = False,
) -> float | None:
    """Return a start offset (seconds) where the clip window has enough material."""
    if not stem_is_valid(path):
        return None

    try:
        audio, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    except (RuntimeError, OSError, ValueError):
        return None
    if audio.size == 0:
        return None

    sr = int(file_sr) if file_sr else SAMPLE_RATE
    mono = np.mean(np.asarray(audio, dtype=np.float32), axis=1)
    n = int(mono.shape[0])
    clip_frames = int(clip_seconds * sr)
    if n < clip_frames:
        return None

    activity_hop = max(1, int(activity_hop_seconds * sr))
    n_hops = n // activity_hop
    hops_in_clip = max(1, int(round(clip_seconds / activity_hop_seconds)))
    if n_hops < hops_in_clip:
        return None

    segment = mono[: n_hops * activity_hop].reshape(n_hops, activity_hop)
    hop_mean_sq = np.mean(np.square(segment, dtype=np.float64), axis=1)
    hop_active = (np.sqrt(hop_mean_sq) >= min_rms).astype(np.int32)
    cum_active = np.concatenate(([0], np.cumsum(hop_active)))
    cum_mean_sq = np.concatenate(([0.0], np.cumsum(hop_mean_sq)))

    search_hop_hops = max(1, int(round(hop_seconds / activity_hop_seconds)))
    first_hop = int(min_start_seconds / activity_hop_seconds)
    last_start_hop = n_hops - hops_in_clip

    best_start: float | None = None
    best_fraction = -1.0

    for start_hop in range(first_hop, last_start_hop + 1, search_hop_hops):
        end_hop = start_hop + hops_in_clip
        window_ms = (cum_mean_sq[end_hop] - cum_mean_sq[start_hop]) / hops_in_clip
        if float(np.sqrt(window_ms)) < min_rms:
            continue
        fraction = float(cum_active[end_hop] - cum_active[start_hop]) / hops_in_clip
        if fraction < min_active_fraction:
            continue
        start_seconds = start_hop * activity_hop / sr
        if not prefer_densest:
            return start_seconds
        if fraction > best_fraction:
            best_fraction = fraction
            best_start = start_seconds

    return best_start


def waveform_to_mono_numpy(
    waveform: torch.Tensor | np.ndarray,
    *,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Convert a stem waveform to mono float32 numpy."""
    del sample_rate
    if isinstance(waveform, torch.Tensor):
        array = waveform.detach().cpu().numpy()
    else:
        array = np.asarray(waveform, dtype=np.float32)
    if array.ndim == 1:
        return array.astype(np.float32, copy=False)
    if array.ndim == 2:
        if array.shape[0] == 1:
            return array[0].astype(np.float32, copy=False)
        return array.mean(axis=0).astype(np.float32, copy=False)
    raise ValueError(f"Expected 1D or 2D waveform, got shape {array.shape}")


def detect_onset_times(
    waveform: np.ndarray,
    *,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Return onset times in seconds."""
    if waveform.size == 0:
        return np.array([], dtype=np.float64)
    onsets = librosa.onset.onset_detect(
        y=waveform,
        sr=sample_rate,
        units="time",
        backtrack=True,
    )
    return np.asarray(onsets, dtype=np.float64)


def load_diverse_stems_manifest(path: Path) -> list[dict]:
    with open(path) as f:
        doc = yaml.safe_load(f) or {}
    return list(doc.get("stems") or [])


def clips_dataset_ready(clips_dir: Path) -> bool:
    return (
        clips_dir.is_dir()
        and (clips_dir / f"{DATA_DIR_NAME}.csv").is_file()
        and (clips_dir / f"{STEMS_FILE_NAME}.csv").is_file()
    )


def resolve_sweep_clips_dir(sweep_dir: Path) -> Path | None:
    """Return the clip dataset directory for a sweep phase output tree."""
    sweep_dir = sweep_dir.resolve()
    clips_dir = sweep_dir / "clips"
    if clips_dataset_ready(clips_dir):
        return clips_dir.resolve()

    marker = sweep_dir / CLIPS_SOURCE_MARKER
    if marker.is_file():
        relative_target = marker.read_text().strip()
        if relative_target:
            target = (sweep_dir / relative_target).resolve()
            if clips_dataset_ready(target):
                return target
    return None


def expose_clips_dir(*, output_dir: Path, source_clips: Path) -> Path:
    """Expose source_clips under output_dir for listening; returns source_clips."""
    source_clips = source_clips.resolve()
    if not source_clips.is_dir():
        raise FileNotFoundError(f"Clip source is not a directory: {source_clips}")

    output_dir.mkdir(parents=True, exist_ok=True)
    dest_clips = output_dir / "clips"
    marker = output_dir / CLIPS_SOURCE_MARKER

    if dest_clips.is_symlink() or (dest_clips.exists() and not clips_dataset_ready(dest_clips)):
        dest_clips.unlink(missing_ok=True)

    relative_target = os.path.relpath(source_clips, output_dir.resolve())

    if not dest_clips.exists():
        try:
            os.symlink(relative_target, dest_clips, target_is_directory=True)
        except OSError:
            pass

    if clips_dataset_ready(dest_clips):
        marker.unlink(missing_ok=True)
        return source_clips

    dest_clips.unlink(missing_ok=True)
    marker.write_text(f"{relative_target}\n")
    return source_clips
