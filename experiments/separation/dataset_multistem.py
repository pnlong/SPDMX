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

# Fixed crop fractions for deterministic val (avoid silent intros).
_VAL_CROP_FRACS = (0.0, 0.25, 0.5, 0.75, 1.0)


def parse_tracks(raw: object) -> list[int]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    text = str(raw).strip()
    if not text:
        return []
    return [int(p) for p in text.replace(",", "|").split("|") if p.strip()]


def _rms(x: np.ndarray, eps: float = 1e-8) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)) + eps))


def energy_ratio(mix_c: np.ndarray, stem_c: np.ndarray, eps: float = 1e-6) -> float:
    """``rms(stem) / rms(mix)``; 0 when the mix is near-silent."""
    # Use raw mean-square (no floor) so all-zero mixes count as silent.
    mix_ms = float(np.mean(np.square(mix_c, dtype=np.float64)))
    if mix_ms < eps * eps:
        return 0.0
    return _rms(stem_c) / _rms(mix_c)


def val_crop_starts(n: int, segment: int) -> list[int]:
    """Deterministic crop offsets spanning the song."""
    if n <= segment:
        return [0]
    max_start = n - segment
    return sorted({int(round(f * max_start)) for f in _VAL_CROP_FRACS})


class StemOtherDataset(Dataset):
    """Load mix + one stem; ``other = mix - stem`` (linearly summable packs).

    Manifest columns: ``song_id``, ``path``, ``mix``, ``tracks``, …
    Audio is read from the chunked SPDMX release (no remapped packs).

    Missing on-disk mixes/stems (stale manifests after a re-chunk) are skipped
    by resampling another index so DataLoader workers do not crash.

    When ``stem_min_energy_ratio > 0``, track/crop pairs with
    ``rms(stem) / rms(mix)`` below the threshold are rejected (train: resample
    another song/crop; val: search tracks × crop grid, else best-energy crop).

    When ``rebuild_mix_from_stems`` is True (default), the mix is the sum of
    listed stems so ``other = mix - stem`` stays exact even if ``mix.flac``
    drifted.
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
        rebuild_mix_from_stems: bool = True,
    ) -> None:
        self.rows = pd.read_csv(manifest_csv)
        self.spdmx_root = Path(spdmx_root)
        self.sample_rate = sample_rate
        self.segment = int(segment_seconds * sample_rate)
        self.channels = channels
        self.train = train
        self.max_resample_tries = max(1, int(max_resample_tries))
        self.stem_min_energy_ratio = float(stem_min_energy_ratio)
        self.rebuild_mix_from_stems = bool(rebuild_mix_from_stems)

    def __len__(self) -> int:
        return len(self.rows)

    def _energy_ok(self, mix_c: np.ndarray, stem_c: np.ndarray) -> bool:
        if self.stem_min_energy_ratio <= 0.0:
            return True
        return energy_ratio(mix_c, stem_c) >= self.stem_min_energy_ratio

    def _load_stem(
        self, row: pd.Series, *, song_id: str, track: int,
    ) -> np.ndarray | None:
        stem_path = resolve_spdmx_stem(
            self.spdmx_root, row, song_id=song_id, track=track,
        )
        if stem_path is None:
            return None
        try:
            stem, _ = load_mono(stem_path, sample_rate=self.sample_rate)
        except (OSError, ValueError, RuntimeError):
            return None
        return stem

    def _sum_stems(
        self, row: pd.Series, *, song_id: str, tracks: list[int],
    ) -> tuple[np.ndarray, dict[int, np.ndarray]] | None:
        parts: dict[int, np.ndarray] = {}
        for t in tracks:
            audio = self._load_stem(row, song_id=song_id, track=int(t))
            if audio is None:
                return None
            parts[int(t)] = audio
        n = min(s.shape[0] for s in parts.values())
        if n <= 0:
            return None
        mix = np.zeros(n, dtype=np.float32)
        for t, s in parts.items():
            clipped = s[:n]
            parts[t] = clipped
            mix += clipped
        return mix, parts

    def _load_mix_flac(self, row: pd.Series, *, song_id: str) -> np.ndarray | None:
        mix_path = resolve_spdmx_mix(self.spdmx_root, row, song_id=song_id)
        if mix_path is None:
            return None
        try:
            mix, _ = load_mono(mix_path, sample_rate=self.sample_rate)
        except (OSError, ValueError, RuntimeError):
            return None
        return mix

    def _resolve_mix_and_stem(
        self,
        row: pd.Series,
        *,
        track: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        song_id = str(row["song_id"])
        tracks = parse_tracks(row.get("tracks"))
        if len(tracks) < 2:
            return None

        if track is None:
            track = int(np.random.choice(tracks)) if self.train else int(tracks[0])
        track = int(track)
        if track not in tracks:
            return None

        if self.rebuild_mix_from_stems:
            summed = self._sum_stems(row, song_id=song_id, tracks=tracks)
            if summed is None:
                return None
            mix, parts = summed
            return mix, parts[track]

        stem = self._load_stem(row, song_id=song_id, track=track)
        if stem is None:
            return None
        mix = self._load_mix_flac(row, song_id=song_id)
        if mix is None:
            summed = self._sum_stems(row, song_id=song_id, tracks=tracks)
            if summed is None:
                return None
            mix, parts = summed
            stem = parts[track]
        n = min(mix.shape[0], stem.shape[0])
        if n <= 0:
            return None
        return mix[:n], stem[:n]

    def _pad_crop(self, mix_c: np.ndarray, stem_c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if mix_c.shape[0] < self.segment:
            pad = self.segment - mix_c.shape[0]
            mix_c = np.pad(mix_c, (0, pad))
            stem_c = np.pad(stem_c, (0, pad))
        return mix_c, stem_c

    def _crop_at(
        self, mix: np.ndarray, stem: np.ndarray, start: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        end = start + self.segment
        return self._pad_crop(mix[start:end], stem[start:end])

    def _crop_pair(self, mix: np.ndarray, stem: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = mix.shape[0]
        if self.train:
            start = int(np.random.randint(0, n - self.segment + 1)) if n > self.segment else 0
        else:
            start = 0
        return self._crop_at(mix, stem, start)

    def _load_pair(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        n = len(self.rows)
        for attempt in range(self.max_resample_tries):
            j = int(idx) if attempt == 0 else int(np.random.randint(0, n))
            row = self.rows.iloc[j]
            pair = self._resolve_mix_and_stem(row)
            if pair is not None:
                return pair
        zeros = np.zeros(self.segment, dtype=np.float32)
        return zeros, zeros.copy()

    def _val_select_crop(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Deterministic energy-aware (track, crop); else best energy ratio."""
        row = self.rows.iloc[int(idx) % len(self.rows)]
        song_id = str(row["song_id"])
        tracks = parse_tracks(row.get("tracks"))
        if len(tracks) < 2:
            zeros = np.zeros(self.segment, dtype=np.float32)
            return zeros, zeros.copy()

        best: tuple[float, np.ndarray, np.ndarray] | None = None

        if self.rebuild_mix_from_stems:
            summed = self._sum_stems(row, song_id=song_id, tracks=tracks)
            if summed is None:
                zeros = np.zeros(self.segment, dtype=np.float32)
                return zeros, zeros.copy()
            mix, parts = summed
            track_iter = [(t, parts[t]) for t in tracks]
        else:
            mix = self._load_mix_flac(row, song_id=song_id)
            track_iter_list: list[tuple[int, np.ndarray]] = []
            for t in tracks:
                stem = self._load_stem(row, song_id=song_id, track=int(t))
                if stem is not None:
                    track_iter_list.append((int(t), stem))
            if mix is None and track_iter_list:
                n = min(s.shape[0] for _, s in track_iter_list)
                mix = np.zeros(n, dtype=np.float32)
                for i, (t, s) in enumerate(track_iter_list):
                    clipped = s[:n]
                    track_iter_list[i] = (t, clipped)
                    mix += clipped
            track_iter = track_iter_list

        if mix is None or not track_iter:
            zeros = np.zeros(self.segment, dtype=np.float32)
            return zeros, zeros.copy()

        for _, stem in track_iter:
            n = min(mix.shape[0], stem.shape[0])
            if n <= 0:
                continue
            mix_n, stem_n = mix[:n], stem[:n]
            for start in val_crop_starts(n, self.segment):
                mix_c, stem_c = self._crop_at(mix_n, stem_n, start)
                ratio = energy_ratio(mix_c, stem_c)
                if self._energy_ok(mix_c, stem_c):
                    return mix_c, stem_c
                if best is None or ratio > best[0]:
                    best = (ratio, mix_c, stem_c)

        if best is not None:
            return best[1], best[2]
        zeros = np.zeros(self.segment, dtype=np.float32)
        return zeros, zeros.copy()

    def _pack(self, mix_c: np.ndarray, stem_c: np.ndarray) -> dict[str, torch.Tensor]:
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

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if not self.train:
            mix_c, stem_c = self._val_select_crop(idx)
            return self._pack(mix_c, stem_c)

        mix_c: np.ndarray | None = None
        stem_c: np.ndarray | None = None
        for attempt in range(self.max_resample_tries):
            j = int(idx) if attempt == 0 else int(np.random.randint(0, len(self.rows)))
            mix, stem = self._load_pair(j)
            mix_c, stem_c = self._crop_pair(mix, stem)
            if self._energy_ok(mix_c, stem_c):
                break
        assert mix_c is not None and stem_c is not None
        return self._pack(mix_c, stem_c)
