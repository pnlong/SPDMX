"""Dataset: random stem vs residual accompaniment from multi-stem SPDMX songs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from experiments.separation.audio_io import load_mono, mono_to_stereo
from experiments.separation.spdmx_io import resolve_spdmx_mix, resolve_spdmx_stem


STEM_OTHER_SOURCES = ("stem", "other")


def parse_tracks(raw: object) -> list[int]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    text = str(raw).strip()
    if not text:
        return []
    return [int(p) for p in text.replace(",", "|").split("|") if p.strip()]


def _rms(x: np.ndarray, eps: float = 1e-8) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)) + eps))


class StemOtherDataset(Dataset):
    """Load mix + one stem; ``other = mix - stem`` (linearly summable packs).

    Manifest columns: ``song_id``, ``path``, ``mix``, ``tracks``, …
    Audio is read from the chunked SPDMX release (no remapped packs).

    Missing on-disk mixes/stems (stale manifests after a re-chunk) are skipped
    by resampling another index so DataLoader workers do not crash.

    When ``train=True`` and ``stem_min_energy_ratio > 0``, track/crop pairs with
    ``rms(stem) / rms(mix)`` below the threshold are resampled so near-silent
    stems do not teach ``other ≈ mix``.
    """

    def __init__(
        self,
        manifest_csv: Path,
        spdmx_root: Path,
        *,
        sample_rate: int = 44100,
        segment_seconds: float = 10.0,
        channels: int = 2,
        train: bool = True,
        max_resample_tries: int = 16,
        stem_min_energy_ratio: float = 0.0,
    ) -> None:
        self.rows = pd.read_csv(manifest_csv)
        self.spdmx_root = Path(spdmx_root)
        self.sample_rate = sample_rate
        self.segment = int(segment_seconds * sample_rate)
        self.channels = channels
        self.train = train
        self.max_resample_tries = max(1, int(max_resample_tries))
        self.stem_min_energy_ratio = float(stem_min_energy_ratio)

    def __len__(self) -> int:
        return len(self.rows)

    def _energy_ok(self, mix_c: np.ndarray, stem_c: np.ndarray) -> bool:
        if self.stem_min_energy_ratio <= 0.0 or not self.train:
            return True
        mix_rms = _rms(mix_c)
        if mix_rms < 1e-6:
            return False
        return (_rms(stem_c) / mix_rms) >= self.stem_min_energy_ratio

    def _resolve_mix_and_stem(
        self, row: pd.Series,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        song_id = str(row["song_id"])
        tracks = parse_tracks(row.get("tracks"))
        if len(tracks) < 2:
            return None

        track = int(np.random.choice(tracks)) if self.train else int(tracks[0])
        stem_path = resolve_spdmx_stem(
            self.spdmx_root, row, song_id=song_id, track=track,
        )
        if stem_path is None:
            return None
        try:
            stem, _ = load_mono(stem_path, sample_rate=self.sample_rate)
        except (OSError, ValueError, RuntimeError):
            return None

        mix_path = resolve_spdmx_mix(self.spdmx_root, row, song_id=song_id)
        mix: np.ndarray | None = None
        if mix_path is not None:
            try:
                mix, _ = load_mono(mix_path, sample_rate=self.sample_rate)
            except (OSError, ValueError, RuntimeError):
                mix = None

        # Fallback: rebuild mix from all listed stems when mix.flac is missing.
        if mix is None:
            parts: list[np.ndarray] = []
            for t in tracks:
                p = resolve_spdmx_stem(self.spdmx_root, row, song_id=song_id, track=int(t))
                if p is None:
                    return None
                try:
                    audio, _ = load_mono(p, sample_rate=self.sample_rate)
                except (OSError, ValueError, RuntimeError):
                    return None
                parts.append(audio)
            n = min(s.shape[0] for s in parts)
            if n <= 0:
                return None
            mix = np.zeros(n, dtype=np.float32)
            for s in parts:
                mix += s[:n]
            stem = stem[:n]

        n = min(mix.shape[0], stem.shape[0])
        if n <= 0:
            return None
        return mix[:n], stem[:n]

    def _load_pair(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        n = len(self.rows)
        for attempt in range(self.max_resample_tries):
            j = int(idx) if attempt == 0 else int(np.random.randint(0, n))
            row = self.rows.iloc[j]
            pair = self._resolve_mix_and_stem(row)
            if pair is not None:
                return pair
        # Silence fallback keeps training alive if a whole batch hits holes.
        zeros = np.zeros(self.segment, dtype=np.float32)
        return zeros, zeros.copy()

    def _crop_pair(self, mix: np.ndarray, stem: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = mix.shape[0]
        if self.train:
            start = int(np.random.randint(0, n - self.segment + 1)) if n > self.segment else 0
        else:
            start = 0
        end = start + self.segment
        mix_c = mix[start:end]
        stem_c = stem[start:end]
        if mix_c.shape[0] < self.segment:
            pad = self.segment - mix_c.shape[0]
            mix_c = np.pad(mix_c, (0, pad))
            stem_c = np.pad(stem_c, (0, pad))
        return mix_c, stem_c

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        mix_c: np.ndarray | None = None
        stem_c: np.ndarray | None = None
        tries = self.max_resample_tries if self.train else 1
        for attempt in range(tries):
            # On energy reject, resample song (and track) as well as crop.
            j = int(idx) if attempt == 0 else int(np.random.randint(0, len(self.rows)))
            mix, stem = self._load_pair(j)
            mix_c, stem_c = self._crop_pair(mix, stem)
            if self._energy_ok(mix_c, stem_c):
                break
        assert mix_c is not None and stem_c is not None

        other_c = mix_c - stem_c
        stacked = np.stack([stem_c, other_c], axis=0)
        peak = float(np.max(np.abs(mix_c))) if mix_c.size else 0.0
        if peak > 1.0:
            mix_c = mix_c / peak
            stacked = stacked / peak

        if self.channels == 2:
            mix_t = torch.from_numpy(mono_to_stereo(mix_c))
            src_t = torch.stack(
                [torch.from_numpy(mono_to_stereo(stacked[i])) for i in range(2)],
                dim=0,
            )
        else:
            mix_t = torch.from_numpy(mix_c).unsqueeze(0)
            src_t = torch.from_numpy(stacked).unsqueeze(1)
        return {"mix": mix_t.float(), "sources": src_t.float()}
