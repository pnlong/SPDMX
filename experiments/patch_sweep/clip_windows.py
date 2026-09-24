"""Pick content-rich fixed-length windows for patch sweep listening clips."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from experiments.listening_shared.clips import (
    DEFAULT_CLIP_SECONDS,
    DEFAULT_MIN_RMS,
    clip_stem_waveform,
    detect_onset_times,
    waveform_to_mono_numpy,
)
from synthesis.audio import stem_duration_seconds, stem_is_valid, stem_n_samples
from shared.config import SAMPLE_RATE

DEFAULT_EDGE_FRACTION = 0.15
DEFAULT_HOP_SECONDS = 1.0
DEFAULT_MIN_SEPARATION_SECONDS = 5.0
ACTIVE_AMP_THRESHOLD = 0.01


def score_clip_window(
    path: Path,
    start_seconds: float,
    *,
    clip_seconds: float = DEFAULT_CLIP_SECONDS,
    min_rms: float = DEFAULT_MIN_RMS,
    edge_fraction: float = DEFAULT_EDGE_FRACTION,
) -> float:
    """Higher is better; returns -inf for inaudible windows."""
    if not stem_is_valid(path):
        return float("-inf")

    duration = stem_duration_seconds(path)
    waveform = clip_stem_waveform(
        path,
        clip_seconds=clip_seconds,
        start_seconds=start_seconds,
    )
    if waveform.size == 0:
        return float("-inf")

    rms = float(np.sqrt(np.mean(np.square(waveform))))
    if rms < min_rms:
        return float("-inf")

    edge_seconds = max(clip_seconds, duration * edge_fraction)
    edge_penalty = 0.0
    if start_seconds < edge_seconds:
        edge_penalty += 0.35 * (1.0 - start_seconds / edge_seconds)
    end_seconds = start_seconds + clip_seconds
    if end_seconds > duration - edge_seconds:
        tail = max(duration - end_seconds, 0.0)
        edge_penalty += 0.35 * (1.0 - tail / edge_seconds)

    mono = waveform_to_mono_numpy(waveform)
    active_fraction = float(np.mean(np.abs(mono) >= ACTIVE_AMP_THRESHOLD))
    onsets = detect_onset_times(mono)
    onset_density = len(onsets) / max(clip_seconds, 1e-6)

    return (rms * 2.0) + active_fraction + (onset_density * 0.25) - edge_penalty


def find_content_rich_clips(
    path: Path,
    *,
    n_clips: int = 3,
    clip_seconds: float = DEFAULT_CLIP_SECONDS,
    min_rms: float = DEFAULT_MIN_RMS,
    hop_seconds: float = DEFAULT_HOP_SECONDS,
    min_separation_seconds: float = DEFAULT_MIN_SEPARATION_SECONDS,
    show_progress: bool = False,
) -> list[float]:
    """Return up to n non-overlapping clip start times (seconds), best content first."""
    if not stem_is_valid(path) or n_clips <= 0:
        return []

    clip_frames = int(clip_seconds * SAMPLE_RATE)
    total_frames = stem_n_samples(path)
    if total_frames < clip_frames:
        return []

    hop_frames = max(1, int(hop_seconds * SAMPLE_RATE))
    scored: list[tuple[float, float]] = []
    hop_range = range(0, total_frames - clip_frames + 1, hop_frames)
    if show_progress:
        from tqdm import tqdm

        hop_range = tqdm(
            hop_range,
            desc=f"Scoring {path.name}",
            unit="hop",
            leave=False,
        )
    for start_frame in hop_range:
        start_seconds = start_frame / SAMPLE_RATE
        score = score_clip_window(
            path,
            start_seconds,
            clip_seconds=clip_seconds,
            min_rms=min_rms,
        )
        if score > float("-inf"):
            scored.append((start_seconds, score))

    scored.sort(key=lambda item: item[1], reverse=True)

    selected: list[float] = []
    for start_seconds, _score in scored:
        if all(
            abs(start_seconds - chosen) >= min_separation_seconds
            for chosen in selected
        ):
            selected.append(start_seconds)
        if len(selected) >= n_clips:
            break

    return sorted(selected)


def clip_id_for(stem_id: str, clip_index: int) -> str:
    return f"{stem_id}_c{clip_index}"


def clip_output_filename(track: int, clip_index: int, audio_format: str) -> str:
    base = f"stem_{track}"
    return f"{base}_c{clip_index}.{audio_format}"
