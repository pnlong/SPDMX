"""SPDMX Torch Dataset for stream-music-gen.

Mirrors :class:`Slakh2100` so the upstream DAC / RMS / mixdown dump scripts
work unchanged. Reads the JSONL index produced by
``experiments.streamgen.prepare_spdmx_index``.

Expected layout under ``root_dir`` (usually ``stream_music_gen_data/spdmx``)::

    spdmx/
      spdmx_multitrack.jsonl   # or set via index_path=
      audio -> /deepfreeze/.../SPDMX   # symlink to release root
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset
from tqdm import tqdm

from stream_music_gen.constants import (
    inst_name_to_inst_class_id,
    midi_prog_to_inst_class_id,
)

logger = logging.getLogger(__name__)


def _default_spdmx_audio_root() -> Path:
    return Path(
        os.environ.get(
            "SPDMX_DATASET_ROOT",
            "/deepfreeze/share/SPDMX/SPDMX",
        )
    )


def _gm_program_to_class_id(program: Optional[int], is_drum: bool) -> int:
    if is_drum:
        return inst_name_to_inst_class_id("Drums")
    if program is None:
        return inst_name_to_inst_class_id("Unknown")
    # GM program in our JSONL is 0-indexed; upstream mapper expects 1–128.
    return midi_prog_to_inst_class_id(int(program) + 1)


class Spdmx(Dataset):
    """SPDMX multitrack stems, indexed by JSONL."""

    VERSION = "1.0.0"
    # Release FLACs are 44.1 kHz; DAC extract resamples → 32 kHz.
    SAMPLE_RATE = 44100
    AUDIO_LINK_NAME = "audio"
    DEFAULT_INDEX_NAME = "spdmx_multitrack.jsonl"

    def __init__(
        self,
        root_dir: str = "stream_music_gen_data/spdmx",
        download: bool = True,
        split: str = "train",
        target_sample_rate: int = 16000,
        include_mix: bool = False,
        load_track: bool = False,
        runtime_transform: callable = None,
        regenerate_metadata: bool = False,
        index_path: Optional[str] = None,
        audio_root: Optional[str] = None,
    ) -> None:
        self.root_dir = Path(root_dir).expanduser().resolve()
        self.download = download  # unused; kept for Slakh-compatible signature
        self.split = split
        self.target_sample_rate = (
            target_sample_rate if target_sample_rate > 0 else self.SAMPLE_RATE
        )
        self.runtime_transform = runtime_transform
        self.include_mix = include_mix
        self.load_track = load_track

        if self.split not in ["train", "test", "validation"]:
            raise ValueError(
                "`split` must be one of ['train', 'test', 'validation']."
            )

        self.index_path = Path(
            index_path
            if index_path is not None
            else self.root_dir / self.DEFAULT_INDEX_NAME
        )
        if not self.index_path.is_file():
            raise FileNotFoundError(
                f"SPDMX index not found: {self.index_path}. "
                "Run: uv run python -m experiments.streamgen.prepare_spdmx_index "
                "and symlink the JSONL here (see prepare_streamgen_data.sh)."
            )

        link = self.root_dir / self.AUDIO_LINK_NAME
        if audio_root is not None:
            self.base_dir = Path(audio_root).expanduser().resolve()
        elif link.exists():
            self.base_dir = link.resolve()
        else:
            self.base_dir = _default_spdmx_audio_root().resolve()
        if not self.base_dir.is_dir():
            raise FileNotFoundError(
                f"SPDMX audio root missing: {self.base_dir}. "
                f"Create {link} → SPDMX release, or set SPDMX_DATASET_ROOT."
            )

        self.resample_transform = T.Resample(
            self.SAMPLE_RATE, self.target_sample_rate
        )

        logger.info(
            "SPDMX split=%s index=%s audio_root=%s",
            self.split,
            self.index_path,
            self.base_dir,
        )
        self.all_metadata = self._load_metadata(regenerate_metadata)
        if self.load_track:
            self.all_metadata = (
                self.all_metadata.groupby("track_name").agg(list).reset_index()
            )

    def _rel_audio_path(self, absolute: str) -> Optional[str]:
        if not absolute:
            return None
        path = Path(absolute)
        try:
            return str(path.resolve().relative_to(self.base_dir))
        except ValueError:
            # Path outside audio root — keep absolute under a stable key.
            logger.warning("stem outside SPDMX root, skipping: %s", absolute)
            return None

    def _load_metadata(self, regenerate_metadata: bool = False) -> pd.DataFrame:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = self.root_dir / f"pt_dataset_metadata_{self.split}.parquet"
        if metadata_path.exists() and not regenerate_metadata:
            return pd.read_parquet(metadata_path)

        rows: list[dict] = []
        with self.index_path.open(encoding="utf-8") as f:
            for line in tqdm(f, desc=f"Loading SPDMX index ({self.split})"):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("split") != self.split:
                    continue
                track_name = str(rec["song_id"])
                for stem in rec.get("stems") or []:
                    rel = self._rel_audio_path(stem.get("audio_path") or "")
                    if not rel:
                        continue
                    program = stem.get("program")
                    is_drum = bool(stem.get("is_drum", False))
                    class_id = _gm_program_to_class_id(program, is_drum)
                    rows.append(
                        {
                            "track_name": track_name,
                            "audio_path": rel,
                            "type": "stem",
                            "instrument_name": "Drums"
                            if is_drum
                            else f"program_{program}",
                            "program_num": int(program)
                            if program is not None
                            else None,
                            "instrument_class_id": class_id,
                            "original_sample_rate": self.SAMPLE_RATE,
                            "audio_format": "flac",
                        }
                    )
                if self.include_mix:
                    mix_rel = self._rel_audio_path(rec.get("mix_path") or "")
                    if mix_rel:
                        rows.append(
                            {
                                "track_name": track_name,
                                "audio_path": mix_rel,
                                "type": "mix",
                                "instrument_name": "mix",
                                "program_num": None,
                                "instrument_class_id": None,
                                "original_sample_rate": self.SAMPLE_RATE,
                                "audio_format": "flac",
                            }
                        )

        all_metadata = pd.DataFrame(rows)
        if metadata_path.exists():
            metadata_path.unlink()
        all_metadata.to_parquet(metadata_path)
        print(
            f"Saved SPDMX metadata ({self.split}): {len(all_metadata)} stems → {metadata_path}"
        )
        return all_metadata

    def __len__(self) -> int:
        return len(self.all_metadata)

    def _get_item_audio(self, idx):
        row = self.all_metadata.iloc[idx]
        file_path = str(self.base_dir / row["audio_path"])
        audio = torchaudio.load(file_path, format="FLAC")[0]
        audio = self.resample_transform(audio)
        base_path = str(Path(file_path).relative_to(self.base_dir))
        item = {
            "audio": audio,
            "file_path": file_path,
            "base_path": base_path,
        }
        if self.runtime_transform:
            item = self.runtime_transform(item)
        return item

    def _get_item_track(self, idx):
        row = self.all_metadata.iloc[idx]
        if self.runtime_transform:
            return self.runtime_transform(row)
        raise RuntimeError("Runtime transform is needed for loading track.")

    def __getitem__(self, idx) -> Dict[str, torch.Tensor | str]:
        if self.load_track:
            return self._get_item_track(idx)
        return self._get_item_audio(idx)
