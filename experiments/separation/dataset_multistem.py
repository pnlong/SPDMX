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


class StemOtherDataset(Dataset):
    """Load mix + one stem; ``other = mix - stem`` (linearly summable packs).

    Manifest columns: ``song_id``, ``path``, ``mix``, ``tracks``, …
    Audio is read from the chunked SPDMX release (no remapped packs).
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
    ) -> None:
        self.rows = pd.read_csv(manifest_csv)
        self.spdmx_root = Path(spdmx_root)
        self.sample_rate = sample_rate
        self.segment = int(segment_seconds * sample_rate)
        self.channels = channels
        self.train = train

    def __len__(self) -> int:
        return len(self.rows)

    def _load_pair(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        row = self.rows.iloc[idx]
        song_id = str(row["song_id"])
        tracks = parse_tracks(row.get("tracks"))
        if len(tracks) < 2:
            raise RuntimeError(f"{song_id}: need >=2 tracks, got {tracks}")
        mix_path = resolve_spdmx_mix(self.spdmx_root, row, song_id=song_id)
        if mix_path is None:
            raise FileNotFoundError(f"missing mix for {song_id}")
        mix, _ = load_mono(mix_path, sample_rate=self.sample_rate)
        track = int(np.random.choice(tracks)) if self.train else int(tracks[0])
        stem_path = resolve_spdmx_stem(
            self.spdmx_root, row, song_id=song_id, track=track,
        )
        if stem_path is None:
            raise FileNotFoundError(f"missing stem {track} for {song_id}")
        stem, _ = load_mono(stem_path, sample_rate=self.sample_rate)
        n = min(mix.shape[0], stem.shape[0])
        if n <= 0:
            n = self.segment
            mix = np.zeros(n, dtype=np.float32)
            stem = np.zeros(n, dtype=np.float32)
        else:
            mix = mix[:n]
            stem = stem[:n]
        return mix, stem

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        mix, stem = self._load_pair(idx)
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
