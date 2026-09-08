"""Torch dataset over remapped 4-stem packs listed in a manifest CSV."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from experiments.separation.audio_io import load_mono, mono_to_stereo
from experiments.separation.paths import TARGETS


class StemPackDataset(Dataset):
    def __init__(
        self,
        manifest_csv: Path,
        packs_root: Path,
        *,
        sample_rate: int = 44100,
        segment_seconds: float = 10.0,
        channels: int = 2,
        train: bool = True,
    ) -> None:
        self.rows = pd.read_csv(manifest_csv)
        self.packs_root = Path(packs_root)
        self.sample_rate = sample_rate
        self.segment = int(segment_seconds * sample_rate)
        self.channels = channels
        self.train = train

    def __len__(self) -> int:
        return len(self.rows)

    def _song_dir(self, idx: int) -> Path:
        rel = self.rows.iloc[idx]["path"]
        return self.packs_root / rel

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        song_dir = self._song_dir(idx)
        sources = []
        for t in TARGETS:
            mono, _ = load_mono(song_dir / f"{t}.flac", sample_rate=self.sample_rate)
            sources.append(mono)
        # Truncate / pad to equal length.
        n = min(s.shape[0] for s in sources)
        if n == 0:
            n = self.segment
            sources = [np.zeros(n, dtype=np.float32) for _ in TARGETS]
        else:
            sources = [s[:n] for s in sources]

        if self.train:
            if n > self.segment:
                start = int(np.random.randint(0, n - self.segment + 1))
            else:
                start = 0
            end = start + self.segment
            chunk = []
            for s in sources:
                c = s[start:end]
                if c.shape[0] < self.segment:
                    c = np.pad(c, (0, self.segment - c.shape[0]))
                chunk.append(c)
            sources = chunk
        else:
            # Eval: take leading segment (full-track eval is in eval.py).
            chunk = []
            for s in sources:
                c = s[: self.segment]
                if c.shape[0] < self.segment:
                    c = np.pad(c, (0, self.segment - c.shape[0]))
                chunk.append(c)
            sources = chunk

        stacked = np.stack(sources, axis=0)  # (S, T)
        mix = stacked.sum(axis=0)
        peak = float(np.max(np.abs(mix))) if mix.size else 0.0
        if peak > 1.0:
            mix = mix / peak
            stacked = stacked / peak

        if self.channels == 2:
            mix_t = torch.from_numpy(mono_to_stereo(mix))
            src_t = torch.stack(
                [torch.from_numpy(mono_to_stereo(stacked[i])) for i in range(len(TARGETS))],
                dim=0,
            )  # (S, 2, T)
        else:
            mix_t = torch.from_numpy(mix).unsqueeze(0)
            src_t = torch.from_numpy(stacked).unsqueeze(1)

        return {"mix": mix_t.float(), "sources": src_t.float()}
