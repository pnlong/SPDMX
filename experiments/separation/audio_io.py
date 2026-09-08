"""Audio helpers for remapping and packing separation stems."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def audio_duration_seconds(path: Path) -> float:
    info = sf.info(str(path))
    if info.samplerate <= 0:
        return 0.0
    return float(info.frames) / float(info.samplerate)


def load_mono(path: Path, *, sample_rate: int | None = None) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), always_2d=True)
    mono = audio.mean(axis=1).astype(np.float32)
    if sample_rate is not None and sr != sample_rate:
        import librosa

        mono = librosa.resample(mono, orig_sr=sr, target_sr=sample_rate)
        sr = sample_rate
    return mono, int(sr)


def sum_stems(stems: list[np.ndarray]) -> np.ndarray:
    if not stems:
        raise ValueError("no stems to sum")
    n = max(s.shape[0] for s in stems)
    out = np.zeros(n, dtype=np.float32)
    for s in stems:
        out[: s.shape[0]] += s
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 1.0:
        out /= peak
    return out


def write_flac(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sample_rate, format="FLAC")


def mono_to_stereo(mono: np.ndarray) -> np.ndarray:
    """(T,) → (2, T) float32."""
    mono = np.asarray(mono, dtype=np.float32)
    return np.stack([mono, mono], axis=0)
