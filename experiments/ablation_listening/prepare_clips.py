"""Prepare and validate ablation listening trial clips (8 conditions, per-category stems)."""

from __future__ import annotations

import argparse
import multiprocessing
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from experiments.ablation_listening.conditions import (
    ABLATION_MUSHRA_CONDITIONS,
    DEFAULT_STEMS_PER_CATEGORY,
    REFERENCE_CONDITION,
    STEM_TRIAL_CATEGORIES,
    condition_roots,
    gm_instrument_label,
)
from experiments.ablation_listening.equivalence import (
    detect_equivalences_for_trial,
)
from experiments.ablation_listening.paths import (
    DEFAULT_CLIPS_DIR,
    DEFAULT_MANIFEST,
)
from experiments.listening_shared.clips import (
    DEFAULT_CLIP_SECONDS,
    DEFAULT_MIN_RMS,
    clip_stem_waveform,
    find_audible_clip_start,
    is_silent,
)
from experiments.probe_stems import load_probe_stems
from shared.config import DATA_DIR_NAME, STEMS_FILE_NAME
from synthesis.audio import (
    mixture_path,
    stem_is_valid,
    stem_path,
    write_audio,
)
from synthesis.listening.catalog import default_ablations_dir, song_id_from_path
from synthesis.patches import resolve_probe_category

# Require material across most of the clip (reject silence + end-burst windows).
DEFAULT_MIN_ACTIVE_FRACTION = 0.7


def _find_dense_clip_start(
    source_path: Path,
    *,
    clip_seconds: float,
    min_rms: float = DEFAULT_MIN_RMS,
    min_active_fraction: float = DEFAULT_MIN_ACTIVE_FRACTION,
    prefer_densest: bool = True,
) -> float | None:
    """Find a window with continuous material (optionally the densest one)."""
    return find_audible_clip_start(
        source_path,
        clip_seconds=clip_seconds,
        min_rms=min_rms,
        min_active_fraction=min_active_fraction,
        prefer_densest=prefer_densest,
    )

def _song_dir(root: Path, song_id: str) -> Path:
    return root / DATA_DIR_NAME / song_id


def _detect_format(song_dir: Path) -> str | None:
    for ext in ("mp3", "flac"):
        if mixture_path(song_dir, ext).is_file():
            return ext
        if any(song_dir.glob(f"stem_*.{ext}")):
            return ext
    return None


def stem_has_all_conditions(
    song_id: str,
    track: int,
    roots: dict[str, Path],
) -> bool:
    """True when ``stem_{track}`` exists under every ablation condition."""
    for root in roots.values():
        song_dir = _song_dir(root, song_id)
        audio_format = _detect_format(song_dir)
        if audio_format is None:
            return False
        if not stem_is_valid(stem_path(song_dir, track, audio_format)):
            return False
    return True


def reference_has_audible_clip(
    song_id: str,
    track: int | None,
    roots: dict[str, Path],
    *,
    trial_type: str = "stem",
    clip_seconds: float = DEFAULT_CLIP_SECONDS,
    min_rms: float = DEFAULT_MIN_RMS,
    min_active_fraction: float = DEFAULT_MIN_ACTIVE_FRACTION,
) -> bool:
    """True when the A1 reference has a dense non-silent ``clip_seconds`` window."""
    song_dir = _song_dir(roots[REFERENCE_CONDITION], song_id)
    audio_format = _detect_format(song_dir)
    if audio_format is None:
        return False
    if trial_type == "mixture":
        source_path = mixture_path(song_dir, audio_format)
    else:
        if track is None:
            return False
        source_path = stem_path(song_dir, int(track), audio_format)
    if not source_path.is_file():
        return False
    if is_silent(source_path, min_rms=min_rms):
        return False
    return _find_dense_clip_start(
        source_path,
        clip_seconds=clip_seconds,
        min_rms=min_rms,
        min_active_fraction=min_active_fraction,
        # Screening only needs existence; densest search is for clip writing.
        prefer_densest=False,
    ) is not None


def song_has_all_mixtures(song_id: str, roots: dict[str, Path]) -> bool:
    for root in roots.values():
        song_dir = _song_dir(root, song_id)
        audio_format = _detect_format(song_dir)
        if audio_format is None:
            return False
        if not mixture_path(song_dir, audio_format).is_file():
            return False
    return True


def _stem_row_category(row: pd.Series) -> str | None:
    category = resolve_probe_category(
        program=int(row.get("program", 0) or 0),
        is_drum=bool(row.get("is_drum", False)),
        track_name=row.get("name"),
    )
    if category not in STEM_TRIAL_CATEGORIES:
        return None
    return category


def _audible_worker(payload: tuple) -> tuple[int, bool]:
    """Pool worker: ``(index, song_id, track, ref_root, clip_seconds) → (index, ok)``."""
    index, song_id, track, ref_root, clip_seconds = payload
    roots = {REFERENCE_CONDITION: Path(ref_root)}
    ok = reference_has_audible_clip(
        song_id,
        track,
        roots,
        clip_seconds=float(clip_seconds),
    )
    return int(index), bool(ok)


def _write_trial_worker(payload: tuple) -> dict:
    trial, ablations_dir, clips_dir, clip_seconds = payload
    return write_trial_clips(
        trial,
        Path(ablations_dir),
        Path(clips_dir),
        clip_seconds=float(clip_seconds),
    )


def _pick_audible_candidates(
    ordered: list[dict],
    *,
    ref_root: Path,
    clip_seconds: float,
    need: int,
    jobs: int,
    desc: str,
) -> list[dict]:
    """Return the first ``need`` audible candidates from ``ordered`` (parallel screen)."""
    if need <= 0 or not ordered:
        return []
    selected: list[dict] = []
    jobs = max(1, int(jobs))
    chunk_size = max(jobs * 2, need)
    offset = 0
    while offset < len(ordered) and len(selected) < need:
        chunk = ordered[offset : offset + chunk_size]
        payloads = [
            (i, c["song_id"], c["track"], str(ref_root), clip_seconds)
            for i, c in enumerate(chunk)
        ]
        if jobs == 1 or len(payloads) <= 1:
            flags = [
                reference_has_audible_clip(
                    c["song_id"],
                    c["track"],
                    {REFERENCE_CONDITION: ref_root},
                    clip_seconds=clip_seconds,
                )
                for c in tqdm(chunk, desc=desc, unit="stem", leave=False)
            ]
        else:
            with multiprocessing.Pool(processes=min(jobs, len(payloads))) as pool:
                results = list(
                    tqdm(
                        pool.imap(_audible_worker, payloads, chunksize=1),
                        total=len(payloads),
                        desc=desc,
                        unit="stem",
                        leave=False,
                    )
                )
            by_index = {idx: ok for idx, ok in results}
            flags = [by_index[i] for i in range(len(chunk))]
        for cand, ok in zip(chunk, flags):
            if not ok:
                continue
            selected.append(cand)
            if len(selected) >= need:
                break
        offset += len(chunk)
    return selected


def select_stem_trials(
    ablations_dir: Path,
    *,
    categories: tuple[str, ...] = STEM_TRIAL_CATEGORIES,
    stems_per_category: int = DEFAULT_STEMS_PER_CATEGORY,
    seed: int = 43,
    clip_seconds: float = DEFAULT_CLIP_SECONDS,
    jobs: int = 1,
) -> list[dict]:
    """Pick ``stems_per_category`` eligible stems for each listening category."""
    roots = condition_roots(ablations_dir)
    basic_dir = roots["basic"]
    stems_df = pd.read_csv(basic_dir / f"{STEMS_FILE_NAME}.csv")
    rng = random.Random(seed)

    probe_by_category: dict[str, list[dict]] = {cat: [] for cat in categories}
    for stem in load_probe_stems():
        if stem.get("category") in probe_by_category:
            probe_by_category[stem["category"]].append(stem)

    trials: list[dict] = []
    used_keys: set[tuple[str, int]] = set()
    ref_root = roots[REFERENCE_CONDITION]

    for category in tqdm(categories, desc="Selecting trials", unit="cat"):
        prelim: list[dict] = []
        for _, row in stems_df.iterrows():
            if _stem_row_category(row) != category:
                continue
            song_path = str(row["path"])
            song_id = song_id_from_path(song_path)
            track = int(row["track"])
            key = (song_id, track)
            if key in used_keys:
                continue
            if not stem_has_all_conditions(song_id, track, roots):
                continue
            prelim.append({
                "song_path": song_path,
                "song_id": song_id,
                "track": track,
                "category": category,
                "note": str(row.get("name") or "").strip() or None,
                "program": int(row.get("program", 0) or 0),
                "is_drum": bool(row.get("is_drum", False)),
            })

        preferred_keys = {
            (p["song_id"], int(p["track"]))
            for p in (probe_by_category.get(category) or [])
        }
        preferred = [c for c in prelim if (c["song_id"], c["track"]) in preferred_keys]
        others = [c for c in prelim if (c["song_id"], c["track"]) not in preferred_keys]
        rng.shuffle(preferred)
        rng.shuffle(others)
        ordered = preferred + others

        candidates = _pick_audible_candidates(
            ordered,
            ref_root=ref_root,
            clip_seconds=clip_seconds,
            need=stems_per_category,
            jobs=jobs,
            desc=f"  {category}",
        )

        if len(candidates) < stems_per_category:
            raise RuntimeError(
                f"Need {stems_per_category} stem trials for category {category!r}, "
                f"but only {len(candidates)} audible stems exist under all "
                f"{len(ABLATION_MUSHRA_CONDITIONS)} conditions in {ablations_dir}."
            )

        for index, picked in enumerate(candidates[:stems_per_category]):
            used_keys.add((picked["song_id"], picked["track"]))
            suffix = f"_{index + 1:02d}" if stems_per_category > 1 else ""
            trials.append({
                "id": f"stem_{category}{suffix}",
                "type": "stem",
                "song_id": picked["song_id"],
                "song_path": picked["song_path"],
                "track": picked["track"],
                "category": picked["category"],
                "note": picked.get("note"),
                "program": int(picked.get("program", 0) or 0),
                "is_drum": bool(picked.get("is_drum", False)),
                "gm_instrument": gm_instrument_label(
                    program=int(picked.get("program", 0) or 0),
                    is_drum=bool(picked.get("is_drum", False)),
                ),
            })

    return trials


def select_mixture_trials(
    ablations_dir: Path,
    *,
    count: int = 4,
    seed: int = 43,
) -> list[dict]:
    roots = condition_roots(ablations_dir)
    songs_df = pd.read_csv(roots["basic"] / f"{DATA_DIR_NAME}.csv")
    paths = [
        str(row["path"])
        for _, row in songs_df.iterrows()
        if song_has_all_mixtures(song_id_from_path(str(row["path"])), roots)
        and reference_has_audible_clip(
            song_id_from_path(str(row["path"])),
            None,
            roots,
            trial_type="mixture",
            clip_seconds=DEFAULT_CLIP_SECONDS,
        )
    ]
    if len(paths) < count:
        raise RuntimeError(
            f"Need {count} mixture trials but only {len(paths)} songs have mixtures "
            f"under all conditions (mixtures are optional via synthesis.mix)."
        )
    rng = random.Random(seed)
    rng.shuffle(paths)
    return [
        {
            "id": f"mix_{index + 1:02d}",
            "type": "mixture",
            "song_id": song_id_from_path(song_path),
            "song_path": song_path,
            "track": None,
            "category": None,
        }
        for index, song_path in enumerate(paths[:count])
    ]


def _clip_reference_path(
    trial: dict,
    ablations_dir: Path,
    *,
    clip_seconds: float,
) -> tuple[Path, str, float]:
    roots = condition_roots(ablations_dir)
    ref_root = roots["basic"]
    song_dir = _song_dir(ref_root, trial["song_id"])
    audio_format = _detect_format(song_dir)
    if audio_format is None:
        raise FileNotFoundError(f"No audio in reference song dir: {song_dir}")

    if trial["type"] == "mixture":
        source_path = mixture_path(song_dir, audio_format)
    else:
        source_path = stem_path(song_dir, int(trial["track"]), audio_format)

    if not source_path.is_file():
        raise FileNotFoundError(f"Missing reference audio: {source_path}")

    start = _find_dense_clip_start(
        source_path,
        clip_seconds=clip_seconds,
        min_rms=DEFAULT_MIN_RMS,
    )
    if start is None:
        raise RuntimeError(
            f"No dense audible {clip_seconds}s clip "
            f"(need ≥{DEFAULT_MIN_ACTIVE_FRACTION:.0%} active) in {source_path}"
        )
    return source_path, audio_format, start


def is_silent_from_waveform(waveform: np.ndarray, *, min_rms: float = DEFAULT_MIN_RMS) -> bool:
    if waveform.size == 0:
        return True
    return float(np.sqrt(np.mean(np.square(waveform)))) < min_rms


def write_trial_clips(
    trial: dict,
    ablations_dir: Path,
    clips_dir: Path,
    *,
    clip_seconds: float = DEFAULT_CLIP_SECONDS,
) -> dict:
    roots = condition_roots(ablations_dir)
    _, audio_format, start_seconds = _clip_reference_path(
        trial,
        ablations_dir,
        clip_seconds=clip_seconds,
    )

    out_dir = clips_dir / trial["id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    for condition_id in ABLATION_MUSHRA_CONDITIONS:
        song_dir = _song_dir(roots[condition_id], trial["song_id"])
        if trial["type"] == "mixture":
            source_path = mixture_path(song_dir, audio_format)
        else:
            source_path = stem_path(song_dir, int(trial["track"]), audio_format)
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing audio: {source_path}")
        waveform = clip_stem_waveform(
            source_path,
            clip_seconds=clip_seconds,
            start_seconds=start_seconds,
        )
        if waveform.size == 0:
            raise RuntimeError(f"Empty clip for {trial['id']} / {condition_id}")
        # Only the reference must be audible; other conditions may be silent
        # (e.g. a failed/quiet realify render) and are still written as-is.
        if (
            condition_id == REFERENCE_CONDITION
            and is_silent_from_waveform(waveform)
        ):
            raise RuntimeError(f"Silent clip for {trial['id']} / {condition_id}")

        out_path = out_dir / f"{condition_id}.{audio_format}"
        write_audio(torch.from_numpy(waveform), out_path, audio_format)
        written[condition_id] = str(out_path.relative_to(clips_dir))

    equivalences = detect_equivalences_for_trial(trial, ablations_dir)
    result = {
        **trial,
        "clip_seconds": clip_seconds,
        "clip_start_seconds": start_seconds,
        "audio_format": audio_format,
        "conditions": written,
    }
    if equivalences:
        result["equivalences"] = equivalences
    return result


def annotate_trial_equivalences(
    trial: dict,
    ablations_dir: Path,
    *,
    pdmx_root: Path | None = None,
) -> dict:
    """Add/refresh ``equivalences`` on a trial via ``route_stem``."""
    equivalences = detect_equivalences_for_trial(
        trial,
        ablations_dir,
        pdmx_root=pdmx_root,
    )
    updated = {**trial}
    if equivalences:
        updated["equivalences"] = equivalences
    else:
        updated.pop("equivalences", None)
    return updated


def annotate_manifest_equivalences(
    manifest_path: Path,
    *,
    ablations_dir: Path | None = None,
    pdmx_root: Path | None = None,
    write: bool = True,
) -> dict:
    """Detect donor-copy equivalences for every trial via ``route_stem``."""
    manifest_path = Path(manifest_path)
    with open(manifest_path) as f:
        doc = yaml.safe_load(f) or {}
    root = Path(ablations_dir) if ablations_dir is not None else Path(
        doc.get("ablations_dir") or ""
    )
    if not root.is_dir():
        raise FileNotFoundError(
            f"Ablations dir missing for equivalence annotate: {root}"
        )
    trials = [
        annotate_trial_equivalences(trial, root, pdmx_root=pdmx_root)
        for trial in (doc.get("trials") or [])
    ]
    doc["trials"] = trials
    if write:
        with open(manifest_path, "w") as f:
            yaml.safe_dump(doc, f, sort_keys=False, default_flow_style=False)
    return doc


def build_manifest(
    ablations_dir: Path,
    *,
    seed: int = 43,
    clip_seconds: float = DEFAULT_CLIP_SECONDS,
    stems_per_category: int = DEFAULT_STEMS_PER_CATEGORY,
    include_mixtures: bool = False,
    mixture_count: int = 4,
    jobs: int = 1,
) -> list[dict]:
    trials = select_stem_trials(
        ablations_dir,
        stems_per_category=stems_per_category,
        seed=seed,
        clip_seconds=clip_seconds,
        jobs=jobs,
    )
    if include_mixtures:
        trials = select_mixture_trials(
            ablations_dir, count=mixture_count, seed=seed,
        ) + trials
    return trials


def prepare_clips(
    ablations_dir: Path,
    clips_dir: Path,
    manifest_path: Path,
    *,
    seed: int = 43,
    clip_seconds: float = DEFAULT_CLIP_SECONDS,
    stems_per_category: int = DEFAULT_STEMS_PER_CATEGORY,
    include_mixtures: bool = False,
    jobs: int = 1,
) -> dict:
    ablations_dir = ablations_dir.resolve()
    clips_dir = clips_dir.resolve()
    manifest_path = manifest_path.resolve()
    clips_dir.mkdir(parents=True, exist_ok=True)
    jobs = max(1, int(jobs))

    trials = build_manifest(
        ablations_dir,
        seed=seed,
        clip_seconds=clip_seconds,
        stems_per_category=stems_per_category,
        include_mixtures=include_mixtures,
        jobs=jobs,
    )
    payloads = [
        (trial, str(ablations_dir), str(clips_dir), clip_seconds)
        for trial in trials
    ]
    if jobs == 1 or len(payloads) <= 1:
        prepared = [
            write_trial_clips(
                trial, ablations_dir, clips_dir, clip_seconds=clip_seconds,
            )
            for trial in tqdm(trials, desc="Writing clips", unit="trial")
        ]
    else:
        with multiprocessing.Pool(processes=min(jobs, len(payloads))) as pool:
            prepared = list(
                tqdm(
                    pool.imap(_write_trial_worker, payloads, chunksize=1),
                    total=len(payloads),
                    desc=f"Writing clips ({min(jobs, len(payloads))} workers)",
                    unit="trial",
                )
            )

    doc = {
        "test_id": "ablation_listening_v2_8cond",
        "clip_seconds": clip_seconds,
        "seed": seed,
        "stems_per_category": stems_per_category,
        "include_mixtures": include_mixtures,
        "conditions": list(ABLATION_MUSHRA_CONDITIONS),
        "categories": list(STEM_TRIAL_CATEGORIES),
        "ablations_dir": str(ablations_dir),
        "trials": prepared,
    }
    with open(manifest_path, "w") as f:
        yaml.safe_dump(doc, f, sort_keys=False, default_flow_style=False)
    return doc


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description=(
            "Prepare 10s clips for the 8-condition ablation listening test "
            "(stem trials × listening categories)."
        ),
    )
    parser.add_argument(
        "--ablations-dir",
        default=str(default_ablations_dir()),
        type=Path,
    )
    parser.add_argument("--clips-dir", default=DEFAULT_CLIPS_DIR, type=Path)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST, type=Path)
    parser.add_argument("--seed", default=43, type=int)
    parser.add_argument("--clip-seconds", default=DEFAULT_CLIP_SECONDS, type=float)
    parser.add_argument(
        "--stems-per-category",
        default=DEFAULT_STEMS_PER_CATEGORY,
        type=int,
        help="Stem trials per listening category (default: 2).",
    )
    parser.add_argument(
        "--include-mixtures",
        action="store_true",
        help="Also include mixture trials (requires mixture files under all conditions).",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        default=max(1, multiprocessing.cpu_count() // 2),
        type=int,
        help="Worker processes for audible screening and clip writing.",
    )
    return parser.parse_args(args)


def main(args=None) -> None:
    opts = parse_args(args)
    doc = prepare_clips(
        opts.ablations_dir,
        opts.clips_dir,
        opts.manifest,
        seed=opts.seed,
        clip_seconds=opts.clip_seconds,
        stems_per_category=opts.stems_per_category,
        include_mixtures=opts.include_mixtures,
        jobs=opts.jobs,
    )
    n_stem = sum(1 for t in doc["trials"] if t["type"] == "stem")
    n_mix = sum(1 for t in doc["trials"] if t["type"] == "mixture")
    n_equiv = sum(len(t.get("equivalences") or {}) for t in doc["trials"])
    n_unique_ratings = sum(
        len(ABLATION_MUSHRA_CONDITIONS) - len(t.get("equivalences") or {})
        for t in doc["trials"]
    )
    print(
        f"Prepared {len(doc['trials'])} trials "
        f"({n_stem} stem, {n_mix} mixture) × "
        f"{len(ABLATION_MUSHRA_CONDITIONS)} conditions"
    )
    print(
        f"Donor equivalences: {n_equiv} omitted duplicates "
        f"→ {n_unique_ratings} unique stimuli across trials "
        f"(× {len(('content', 'realism'))} scales)"
    )
    print(f"Manifest: {opts.manifest.resolve()}")
    print(f"Clips: {opts.clips_dir.resolve()}")


if __name__ == "__main__":
    main()
