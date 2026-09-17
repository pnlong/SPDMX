"""Convert SPDMX/Slakh CSV manifests into YourMT3 ``yourmt3_indexes`` file lists.

YourMT3 expects, under ``data_home``::

    yourmt3_indexes/{dataset}_{split}_file_list.json
    {dataset}_yourmt3_16k/{split}/{id}/mix.wav
    {dataset}_yourmt3_16k/{split}/{id}/{id}_notes.npy
    {dataset}_yourmt3_16k/{split}/{id}/{id}_note_events.npy

This script reads our mix+MIDI CSV manifests, writes 16 kHz WAVs + note caches,
and emits the JSON indexes. Mix-only (no stem.npy) — ``has_stems=False`` at train time.

Usage::

    uv run python -m experiments.transcription.manifest_to_yourmt3_indexes
    uv run python -m experiments.transcription.manifest_to_yourmt3_indexes --arms slakh,spdmx
    uv run python -m experiments.transcription.manifest_to_yourmt3_indexes --arms spdmx --max-songs 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio
from tqdm import tqdm

from experiments.transcription.paths import REPO_ROOT, TRANS_DIR
from shared.config import OUTPUT_DIR, DEV_DIR_NAME, EXPERIMENTS_DIR_NAME

PAPER_MANIFESTS = REPO_ROOT / "analysis" / "paper_data" / "transcription_manifests"
YOURMT3_SRC = TRANS_DIR / "YourMT3" / "amt" / "src"
DEFAULT_DATA_HOME = (
    Path(OUTPUT_DIR) / DEV_DIR_NAME / EXPERIMENTS_DIR_NAME / "transcription" / "yourmt3_data"
)

ARM_TO_MANIFEST = {
    "slakh": "manifest_slakh.csv",
    "spdmx": "manifest_spdmx.csv",
    "both": "manifest_both.csv",
    "spdmx_hour_matched": "manifest_spdmx_hour_matched.csv",
}
# YourMT3 dataset_name used in file_list JSON filenames + data_presets.
# Avoid plain "slakh" — YourMT3 already ships a built-in slakh preset.
ARM_TO_DATASET = {
    "slakh": "slakh_redux",
    "spdmx": "spdmx",
    "both": "slakh_spdmx",  # written as one merged index set under this name
    "spdmx_hour_matched": "spdmx_hm",
}

# Old C1 dataset dirs/indexes → current names (absolute paths inside JSONs must move too).
_LEGACY_DATASET_RENAMES = {
    "c1_slakh": "slakh_redux",
    "c1_both": "slakh_spdmx",
}


def _migrate_legacy_c1_names(data_home: Path) -> None:
    """Rename leftover ``c1_*`` audio trees / indexes and rewrite absolute paths."""
    index_dir = data_home / "yourmt3_indexes"
    for old, new in _LEGACY_DATASET_RENAMES.items():
        old_audio = data_home / f"{old}_yourmt3_16k"
        new_audio = data_home / f"{new}_yourmt3_16k"
        if old_audio.is_dir():
            if new_audio.exists():
                print(
                    f"warning: both {old_audio.name} and {new_audio.name} exist; "
                    f"leave {old_audio.name} (delete it yourself if unused)"
                )
            else:
                old_audio.rename(new_audio)
                print(f"renamed {old_audio.name} → {new_audio.name}")

        if index_dir.is_dir():
            for old_json in sorted(index_dir.glob(f"{old}_*_file_list.json")):
                new_json = index_dir / old_json.name.replace(f"{old}_", f"{new}_", 1)
                if new_json.exists():
                    print(f"warning: {new_json.name} already exists; skip renaming {old_json.name}")
                else:
                    old_json.rename(new_json)
                    print(f"renamed {old_json.name} → {new_json.name}")

    # Rewrite absolute paths inside any file_list that still mentions old names.
    if index_dir.is_dir():
        path_keys = ("mix_audio_file", "notes_file", "note_events_file")
        for json_path in sorted(index_dir.glob("*_file_list.json")):
            text = json_path.read_text(encoding="utf-8")
            updated = text
            for old, new in _LEGACY_DATASET_RENAMES.items():
                updated = updated.replace(f"{old}_yourmt3_16k", f"{new}_yourmt3_16k")
            if updated == text:
                continue
            data = json.loads(updated)
            # Keep paths consistent with on-disk location after rename.
            for entry in data.values():
                for key in path_keys:
                    p = entry.get(key)
                    if not isinstance(p, str):
                        continue
                    for old, new in _LEGACY_DATASET_RENAMES.items():
                        p = p.replace(f"{old}_yourmt3_16k", f"{new}_yourmt3_16k")
                    entry[key] = p
            json_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
            print(f"updated paths in {json_path.name}")

    legacy_ready = data_home / "C1_INDEXES_READY.json"
    new_ready = data_home / "INDEXES_READY.json"
    if legacy_ready.is_file() and not new_ready.exists():
        legacy_ready.rename(new_ready)
        print(f"renamed {legacy_ready.name} → {new_ready.name}")
    elif legacy_ready.is_file():
        legacy_ready.unlink()
        print(f"removed stale {legacy_ready.name}")



def _ensure_yourmt3_on_path() -> None:
    if not YOURMT3_SRC.is_dir():
        raise SystemExit(
            f"YourMT3 src missing at {YOURMT3_SRC}. "
            "Run: uv run python -m experiments.transcription.setup_yourmt3"
        )
    src = str(YOURMT3_SRC.resolve())
    if src not in sys.path:
        sys.path.insert(0, src)


def _safe_id(song_id: str) -> str:
    return str(song_id).replace("/", "__").replace(os.sep, "__")


def _convert_mix_to_16k_wav(src: Path, dest: Path) -> int:
    """Write mono 16 kHz PCM wav; return n_frames."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 0:
        info = torchaudio.info(str(dest))
        return int(info.num_frames)
    wav, sr = torchaudio.load(str(src))
    if wav.dim() == 2 and wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    torchaudio.save(str(dest), wav, 16000, encoding="PCM_S", bits_per_sample=16)
    return int(wav.shape[-1])


def _midi_to_notes(mid_path: Path, item_id: str):
    from utils.midi import DRUM_PROGRAM, midi2note
    from utils.note2event import extract_program_from_notes, note2note_event

    try:
        notes, dur_sec = midi2note(
            str(mid_path),
            binary_velocity=True,
            ch_9_as_drum=True,
            force_all_drum=False,
            force_all_program_to=None,
            trim_overlap=True,
            fix_offset=True,
            quantize=True,
            verbose=0,
            minimum_offset_sec=0.01,
            drum_offset_sec=0.01,
            ignore_pedal=False,
        )
    except ValueError:
        # Missing program_change messages — treat as piano.
        notes, dur_sec = midi2note(
            str(mid_path),
            binary_velocity=True,
            ch_9_as_drum=True,
            force_all_program_to=0,
            trim_overlap=True,
            fix_offset=True,
            quantize=True,
            verbose=0,
            minimum_offset_sec=0.01,
            drum_offset_sec=0.01,
            ignore_pedal=False,
        )
    programs = sorted(extract_program_from_notes(notes))
    if not programs:
        programs = [0]
    is_drum = [1 if int(p) == int(DRUM_PROGRAM) else 0 for p in programs]
    notes_dict = {
        "mtrack_id": item_id,
        "program": programs,
        "is_drum": is_drum,
        "duration_sec": float(dur_sec),
        "notes": notes,
    }
    note_events_dict = {
        "mtrack_id": item_id,
        "program": programs,
        "is_drum": is_drum,
        "duration_sec": float(dur_sec),
        "note_events": note2note_event(notes, sort=True, return_activity=True),
    }
    return notes_dict, note_events_dict, programs, is_drum


def convert_manifest(
    manifest_csv: Path,
    *,
    dataset_name: str,
    data_home: Path,
    max_songs: int | None = None,
    skip_existing: bool = True,
) -> dict[str, Path]:
    """Convert one CSV → per-split file_list JSONs. Returns map split→json path."""
    _ensure_yourmt3_on_path()
    df = pd.read_csv(manifest_csv)
    if max_songs is not None:
        # Keep split balance roughly: take first N overall after shuffle by song_id order.
        df = df.head(int(max_songs))

    index_dir = data_home / "yourmt3_indexes"
    index_dir.mkdir(parents=True, exist_ok=True)
    out_jsons: dict[str, Path] = {}

    for split, group in df.groupby(df["split"].astype(str), sort=False):
        file_list: dict[int, dict] = {}
        idx = 0
        for _, row in tqdm(
            group.iterrows(),
            total=len(group),
            desc=f"{dataset_name}/{split}",
        ):
            mix = Path(str(row["mix_path"]))
            midi = Path(str(row["midi_path"]))
            song_id = str(row["song_id"])
            item_id = _safe_id(song_id)
            if not mix.is_file() or not midi.is_file():
                continue

            item_dir = data_home / f"{dataset_name}_yourmt3_16k" / split / item_id
            wav_path = item_dir / "mix.wav"
            notes_path = item_dir / f"{item_id}_notes.npy"
            ne_path = item_dir / f"{item_id}_note_events.npy"

            try:
                if skip_existing and wav_path.is_file() and notes_path.is_file() and ne_path.is_file():
                    n_frames = int(torchaudio.info(str(wav_path)).num_frames)
                    cached = np.load(notes_path, allow_pickle=True).item()
                    programs = list(cached.get("program", [0]))
                    is_drum = list(cached.get("is_drum", [0] * len(programs)))
                else:
                    n_frames = _convert_mix_to_16k_wav(mix, wav_path)
                    notes_dict, note_events_dict, programs, is_drum = _midi_to_notes(
                        midi, item_id
                    )
                    np.save(notes_path, notes_dict, allow_pickle=True, fix_imports=False)
                    np.save(ne_path, note_events_dict, allow_pickle=True, fix_imports=False)
            except Exception as exc:  # pragma: no cover
                print(f"skip {song_id}: {exc}")
                continue

            file_list[idx] = {
                "mtrack_id": item_id,
                "song_id": song_id,
                "n_frames": int(n_frames),
                "mix_audio_file": str(wav_path.resolve()),
                "notes_file": str(notes_path.resolve()),
                "note_events_file": str(ne_path.resolve()),
                "midi_file": str(midi.resolve()),
                "program": [int(p) for p in programs],
                "is_drum": [int(d) for d in is_drum],
                # mix-only: omit stem_file so CachedAudioDataset sets has_stems=False
            }
            idx += 1

        json_path = index_dir / f"{dataset_name}_{split}_file_list.json"
        # JSON keys must be strings for json.dump of int keys — YourMT3 loads and uses .items()
        with json_path.open("w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in file_list.items()}, f, indent=2)
        print(f"wrote {json_path} ({len(file_list)} songs)")
        out_jsons[str(split)] = json_path

    return out_jsons


def _symlink_data_home(data_home: Path) -> Path:
    """Point YourMT3's default ../../data at our data_home."""
    link = TRANS_DIR / "YourMT3" / "data"
    data_home.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.exists():
        if link.resolve() == data_home.resolve():
            return link
        if link.is_symlink() or link.is_file():
            link.unlink()
        elif link.is_dir() and not any(link.iterdir()):
            link.rmdir()
        else:
            print(f"warning: {link} exists and is not our symlink; leave as-is")
            return link
    link.symlink_to(data_home)
    print(f"symlinked {link} → {data_home}")
    return link


_SPDMX_SINGLE_BLOCK = '''
    # --- SPDMX transcription pilots (generated by experiments.transcription.manifest_to_yourmt3_indexes) ---
    "slakh_redux": {
            "eval_vocab": [GM_INSTR_CLASS],
            "eval_drum_vocab": drum_vocab_presets["gm"],
            "dataset_name": "slakh_redux",
            "train_split": "train",
            "validation_split": "validation",
            "test_split": "test",
            "has_stem": False,
    },
    "spdmx": {
            "eval_vocab": [GM_INSTR_CLASS],
            "eval_drum_vocab": drum_vocab_presets["gm"],
            "dataset_name": "spdmx",
            "train_split": "train",
            "validation_split": "validation",
            "test_split": "test",
            "has_stem": False,
    },
    "spdmx_hm": {
            "eval_vocab": [GM_INSTR_CLASS],
            "eval_drum_vocab": drum_vocab_presets["gm"],
            "dataset_name": "spdmx_hm",
            "train_split": "train",
            "validation_split": "validation",
            "test_split": "test",
            "has_stem": False,
    },
    "slakh_spdmx": {
            "eval_vocab": [GM_INSTR_CLASS],
            "eval_drum_vocab": drum_vocab_presets["gm"],
            "dataset_name": "slakh_spdmx",
            "train_split": "train",
            "validation_split": "validation",
            "test_split": "test",
            "has_stem": False,
    },
'''

_SPDMX_MULTI_BLOCK = '''
    "slakh_vs_spdmx": {
        "presets": ["slakh_redux", "spdmx"],
        "weights": [0.5, 0.5],
        "eval_vocab": [None, None],
        "eval_drum_vocab": drum_vocab_presets["gm"],
    },
'''


def _ensure_data_presets() -> None:
    """Idempotently add SPDMX transcription presets to YourMT3 data_presets.py."""
    import re

    path = YOURMT3_SRC / "config" / "data_presets.py"
    text = path.read_text(encoding="utf-8")
    already_ok = (
        '"slakh_redux"' in text
        and '"spdmx"' in text
        and '"slakh_spdmx"' in text
        and '"slakh_vs_spdmx"' in text
        and '"c1_slakh"' not in text
        and '"c1_both"' not in text
    )
    # Orphaned insert sat between the single_cfg close and multi_cfg open.
    orphaned = bool(
        re.search(
            r"\}\s*\n\s*# --- SPDMX (?:C1|transcription) pilots",
            text,
        )
    )
    if already_ok and not orphaned:
        return

    # Remove any previous SPDMX pilot entries (keep the single_cfg closing brace).
    text = re.sub(
        r"\n[ \t]*# --- SPDMX (?:C1|transcription) pilots \(generated by experiments\.transcription\.manifest_to_yourmt3_indexes\) ---"
        r"(?:\n[ \t]*\"[^\"]+\"\s*:\s*\{.*?},?)+",
        "",
        text,
        count=1,
        flags=re.S,
    )
    text = re.sub(
        r'\n[ \t]*"c1_slakh_vs_spdmx"\s*:\s*\{[^{}]*\},?',
        "",
        text,
        count=1,
        flags=re.S,
    )

    multi_marker = "\ndata_preset_multi_cfg = {"
    if multi_marker not in text:
        raise SystemExit(f"could not patch presets in {path}")
    if '"slakh_redux"' not in text:
        idx = text.find(multi_marker)
        close_idx = text.rfind("}", 0, idx)
        if close_idx < 0:
            raise SystemExit(f"could not find end of data_preset_single_cfg in {path}")
        text = text[:close_idx] + _SPDMX_SINGLE_BLOCK + text[close_idx:]

    if '"slakh_vs_spdmx"' not in text:
        text = text.replace(
            "data_preset_multi_cfg = {",
            "data_preset_multi_cfg = {" + _SPDMX_MULTI_BLOCK,
            1,
        )

    path.write_text(text, encoding="utf-8")
    print(f"patched transcription presets into {path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arms",
        type=str,
        default="slakh,spdmx",
        help="Comma-separated arms: slakh,spdmx,both,spdmx_hour_matched",
    )
    parser.add_argument(
        "--data-home",
        type=Path,
        default=None,
        help=f"YourMT3 data_home (default: {DEFAULT_DATA_HOME})",
    )
    parser.add_argument("--max-songs", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="Recompute even if caches exist")
    parser.add_argument("--skip-symlink", action="store_true")
    parser.add_argument("--skip-presets", action="store_true")
    args = parser.parse_args(argv)

    data_home = Path(args.data_home or DEFAULT_DATA_HOME)
    try:
        data_home.mkdir(parents=True, exist_ok=True)
    except OSError:
        data_home = TRANS_DIR / "yourmt3_data"
        data_home.mkdir(parents=True, exist_ok=True)
        print(f"deepfreeze unavailable; using {data_home}")

    if not args.skip_symlink:
        _symlink_data_home(data_home)
    if not args.skip_presets:
        _ensure_data_presets()
    _migrate_legacy_c1_names(data_home)

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    summary = {"data_home": str(data_home), "arms": {}}
    for arm in arms:
        if arm not in ARM_TO_MANIFEST:
            raise SystemExit(f"unknown arm {arm}; choose from {list(ARM_TO_MANIFEST)}")
        manifest = PAPER_MANIFESTS / ARM_TO_MANIFEST[arm]
        if not manifest.is_file():
            raise SystemExit(
                f"missing {manifest}; run: uv run python -m experiments.transcription.prepare_manifest"
            )
        dataset_name = ARM_TO_DATASET[arm]
        jsons = convert_manifest(
            manifest,
            dataset_name=dataset_name,
            data_home=data_home,
            max_songs=args.max_songs,
            skip_existing=not args.force,
        )
        summary["arms"][arm] = {s: str(p) for s, p in jsons.items()}

    summary_path = data_home / "INDEXES_READY.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {summary_path}")
    print()
    print("Train examples:")
    print("  uv run python -m experiments.transcription.train --arm slakh --gpu 1")
    print("  uv run python -m experiments.transcription.train --arm spdmx --gpu 2")


if __name__ == "__main__":
    main()
