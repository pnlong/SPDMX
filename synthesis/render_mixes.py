"""Render full-song mixes under ``SPDMX_dev/mix/<song_id>.flac`` via ffmpeg.

After summable stems exist in ``audio/<song_id>/*.flac``, this sums them with
ffmpeg ``amix`` (``normalize=0`` = raw linear sum) and writes a mono FLAC.

Production path is ``synthesis.final --only-pass mix`` (normalize + this step).
Standalone CLI remains for debugging.

Resumable / dirty-aware: skips songs whose ``mix/<song_id>.flac`` exists and is
at least as new as every ``audio/<song_id>/*.flac`` stem, unless ``force`` /
``force_ids`` / ``synthesis.final --reset``.

Example::

    uv run python -m synthesis.final --only-pass mix -j 32
    uv run python -m synthesis.render_mixes -j 32   # debug only
"""

from __future__ import annotations

import argparse
import multiprocessing
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from analysis.corrected_midi import TRACK_MAP_FILE_NAME
from shared.config import (
    FLAC_AUDIO_FORMAT,
    OUTPUT_DIR,
    SPDMX_AUDIO_DIR_NAME,
    SPDMX_FILE_NAME,
    SPDMX_MIX_DIR_NAME,
)
from synthesis.paths import spdmx_audio_dir, spdmx_dev_dir, spdmx_mix_dir


def parse_args(args=None, namespace=None):
    parser = argparse.ArgumentParser(
        description=(
            "Sum SPDMX_dev/audio stems into SPDMX_dev/mix/<song_id>.flac "
            "(ffmpeg amix, normalize=0, mono). Prefer "
            "`synthesis.final --only-pass mix` in production."
        ),
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        default=OUTPUT_DIR,
        type=str,
        help="Pipeline output root (contains SPDMX_dev/).",
    )
    parser.add_argument(
        "--dataset-dir",
        default=None,
        type=str,
        help="Flat production tree (default: {output_dir}/SPDMX_dev).",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=8,
        help="Parallel ffmpeg workers (default: 8).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-render mixes even when the destination FLAC is up to date.",
    )
    parser.add_argument(
        "--repair-sums",
        action="store_true",
        help=(
            "Delete mix files that are not the sample-wise sum of audio stems, "
            "then remake mixes only (leaves audio/). Prefer "
            "`synthesis.final --only-pass mix --repair-mix-sums` to also "
            "rebuild audio/ from raw/."
        ),
    )
    parser.add_argument(
        "--delete-bad-sums",
        action="store_true",
        help=(
            "Only delete mix files that fail the sample-wise sum check; "
            "re-run mix afterward to remake them."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List songs that would be rendered without writing.",
    )
    return parser.parse_args(args=args, namespace=namespace)


def _stem_flac_paths(song_audio_dir: Path) -> list[Path]:
    """Dense track stems ``0.flac``, ``1.flac``, … (numeric basename only)."""
    if not song_audio_dir.is_dir():
        return []
    stems: list[tuple[int, Path]] = []
    for path in song_audio_dir.iterdir():
        if not path.is_file() or path.suffix.lower() != f".{FLAC_AUDIO_FORMAT}":
            continue
        try:
            track = int(path.stem)
        except ValueError:
            continue
        stems.append((track, path))
    stems.sort(key=lambda item: item[0])
    return [path for _, path in stems]


def _load_mono_float32(path: Path) -> tuple[np.ndarray, int]:
    """Load FLAC as mono float32 waveform ``(n_samples,)`` and sample rate."""
    import soundfile as sf

    audio, sr = sf.read(str(path), always_2d=True, dtype="float32")
    if audio.shape[1] == 1:
        return audio[:, 0], int(sr)
    return audio.mean(axis=1).astype(np.float32, copy=False), int(sr)


def sum_stem_waveforms(stem_paths: list[Path]) -> tuple[np.ndarray, int]:
    """Sample-wise sum of stem FLACs (pad shorter stems; mono)."""
    if not stem_paths:
        raise ValueError("no stems to sum")
    waves: list[np.ndarray] = []
    sr0: int | None = None
    for path in stem_paths:
        wave, sr = _load_mono_float32(path)
        if sr0 is None:
            sr0 = sr
        elif sr != sr0:
            raise ValueError(f"sample-rate mismatch: {path} has {sr}, expected {sr0}")
        waves.append(wave)
    assert sr0 is not None
    n = max(w.shape[0] for w in waves)
    total = np.zeros(n, dtype=np.float64)
    for wave in waves:
        total[: wave.shape[0]] += wave
    return total.astype(np.float32, copy=False), sr0


def mix_sum_failure_is_safe_to_delete(reason: str) -> bool:
    """True only for content mismatches on an existing mix file.

    Never treat missing stems, I/O errors, or missing mixes as deletable — those
    would wipe good mixes during NFS blips or path mistakes.
    """
    text = str(reason)
    # Must be a real on-disk content disagreement.
    if "max|mix-sum|" in text:
        return True
    if "length mix=" in text:
        return True
    if "sample-rate mix=" in text:
        return True
    return False


def mix_matches_stem_sum(
    song_id: str,
    *,
    audio_root: Path | str,
    mix_root: Path | str,
    stem_paths: list[Path] | tuple[Path, ...] | None = None,
    atol: float = 2e-4,
    rtol: float = 1e-4,
    max_len_delta: int = 1,
) -> tuple[str | None, float | None]:
    """Return ``(error|None, song_length_seconds|None)``.

    ``None`` error means ``mix/<id>.flac`` matches ``sum(audio stems)``.
    ``song_length`` is set whenever the mix FLAC fully decodes (even on sum
    mismatch), so callers can persist durations without a second open.
    """
    song_id = str(song_id)
    audio_dir = Path(audio_root) / song_id
    mix_path = Path(mix_root) / f"{song_id}.flac"
    if stem_paths is not None:
        stems = [Path(p) for p in stem_paths]
        for path in stems:
            try:
                if not path.is_file() or path.stat().st_size <= 0:
                    return f"{song_id}: missing stem {path}", None
            except OSError:
                return f"{song_id}: missing stem {path}", None
    else:
        stems = _stem_flac_paths(audio_dir)
    if not stems:
        return f"{song_id}: no audio stems", None
    if not mix_path.is_file() or mix_path.stat().st_size <= 0:
        return f"{song_id}: missing mix {mix_path}", None
    try:
        expected, sr_stems = sum_stem_waveforms(stems)
        mix, sr_mix = _load_mono_float32(mix_path)
    except Exception as exc:  # noqa: BLE001
        return f"{song_id}: read failed ({exc})", None
    duration = float(mix.shape[0] / float(sr_mix)) if sr_mix > 0 else None
    if sr_mix != sr_stems:
        return f"{song_id}: sample-rate mix={sr_mix} stems={sr_stems}", duration
    if abs(mix.shape[0] - expected.shape[0]) > max_len_delta:
        return (
            f"{song_id}: length mix={mix.shape[0]} sum={expected.shape[0]} "
            f"(delta>{max_len_delta})",
            duration,
        )
    n = min(mix.shape[0], expected.shape[0])
    if n <= 0:
        return f"{song_id}: empty audio", duration
    if not np.allclose(mix[:n], expected[:n], rtol=rtol, atol=atol):
        peak = float(np.max(np.abs(mix[:n] - expected[:n])))
        return (
            f"{song_id}: max|mix-sum|={peak:.4g} (atol={atol}, rtol={rtol})",
            duration,
        )
    return None, duration


def _mix_sum_check_job(
    payload: tuple[str, str, str, tuple[str, ...] | None],
) -> tuple[str, str | None, float | None]:
    """Picklable worker: ``(song_id, audio_root, mix_root, stems|None)``."""
    song_id, audio_root, mix_root, stem_strs = payload
    stems = [Path(p) for p in stem_strs] if stem_strs is not None else None
    err, duration = mix_matches_stem_sum(
        song_id,
        audio_root=audio_root,
        mix_root=mix_root,
        stem_paths=stems,
    )
    return song_id, err, duration


def _claimed_stem_paths_by_song(
    media_dir: Path,
    *,
    audio_format: str = FLAC_AUDIO_FORMAT,
) -> dict[str, list[Path]] | None:
    """Map ``song_id`` → claimed ``audio/<id>/<track>.flac`` paths from the track map.

    Returns ``None`` when the production CSV is missing or has no track column
    (caller should fall back to directory listing).
    """
    csv_path = media_dir / f"{SPDMX_FILE_NAME}.csv"
    if not csv_path.is_file() or csv_path.stat().st_size <= 0:
        return None
    table = pd.read_csv(csv_path, low_memory=False)
    if not {"song_id", "track"}.issubset(table.columns):
        return None
    audio_root = media_dir / SPDMX_AUDIO_DIR_NAME
    out: dict[str, list[Path]] = {}
    for song_id, track in zip(
        table["song_id"].astype(str),
        table["track"].tolist(),
        strict=False,
    ):
        path = audio_root / song_id / f"{int(track)}.{audio_format}"
        bucket = out.setdefault(song_id, [])
        if path not in bucket:
            bucket.append(path)
    for song_id, paths in out.items():
        out[song_id] = sorted(paths, key=lambda p: p.name)
    return out


def _song_ids_for_mix_sum_check(
    media_dir: Path,
    *,
    tables_dir: str | Path | None = None,
) -> list[str]:
    audio_root = media_dir / SPDMX_AUDIO_DIR_NAME
    mix_root = media_dir / SPDMX_MIX_DIR_NAME
    if not audio_root.is_dir():
        raise RuntimeError(f"Missing audio tree for mix-sum check: {audio_root}")
    if not mix_root.is_dir():
        raise RuntimeError(f"Missing mix tree for mix-sum check: {mix_root}")

    if tables_dir is not None:
        from synthesis.mix import song_mix_paths_from_media

        mix_paths = song_mix_paths_from_media(media_dir, tables_dir=tables_dir)
        song_ids: list[str] = []
        mix_prefix = str(mix_root).rstrip("/") + "/"
        for p in mix_paths:
            text = str(p).replace("\\", "/")
            if not text.startswith(mix_prefix):
                text = str(Path(p).relative_to(mix_root)).replace("\\", "/")
            else:
                text = text[len(mix_prefix) :]
            if text.endswith(".flac"):
                text = text[: -len(".flac")]
            song_ids.append(text)
        return song_ids
    return song_ids_from_stems_csv(media_dir)


def _mix_sum_payloads(
    media_dir: Path,
    *,
    tables_dir: str | Path | None = None,
) -> list[tuple[str, str, str, tuple[str, ...] | None]]:
    """Build parallel payloads for decode+sum verify."""
    audio_root = media_dir / SPDMX_AUDIO_DIR_NAME
    mix_root = media_dir / SPDMX_MIX_DIR_NAME
    song_ids = _song_ids_for_mix_sum_check(media_dir, tables_dir=tables_dir)
    claimed = _claimed_stem_paths_by_song(media_dir)
    payloads: list[tuple[str, str, str, tuple[str, ...] | None]] = []
    for song_id in song_ids:
        stem_tuple: tuple[str, ...] | None = None
        if claimed is not None and song_id in claimed:
            stem_tuple = tuple(str(p) for p in claimed[song_id])
        payloads.append((song_id, str(audio_root), str(mix_root), stem_tuple))
    return payloads


def collect_mix_sum_mismatches(
    media_dir: str | Path,
    *,
    tables_dir: str | Path | None = None,
    jobs: int = 1,
    limit: int | None = None,
    persist_song_lengths: bool = True,
) -> list[tuple[str, str]]:
    """Return ``[(song_id, reason), ...]`` for mixes that fail decode or sum check.

    When ``limit`` is set, stop after that many mismatches (verify fail-fast).
    Each song is fully decoded once (stems + mix) while checking the sum.
    Decoded mix durations are written to ``stems.csv`` ``song_length`` when
    ``persist_song_lengths`` is True (skips a later songs-table header pass).
    """
    root = Path(media_dir)
    payloads = _mix_sum_payloads(root, tables_dir=tables_dir)
    if not payloads:
        raise RuntimeError(f"No songs to mix-sum check under {media_dir}")

    n_jobs = max(1, int(jobs))
    label = (
        f"verify decode+mix=sum (-j {n_jobs})"
        if n_jobs > 1
        else "verify decode+mix=sum"
    )
    bad: list[tuple[str, str]] = []
    lengths: dict[str, float] = {}
    stop_at = int(limit) if limit is not None else None

    def _maybe_stop() -> bool:
        return stop_at is not None and len(bad) >= stop_at

    def _handle(sid: str, err: str | None, duration: float | None) -> None:
        if duration is not None and duration > 0:
            lengths[sid] = float(duration)
        if err:
            bad.append((sid, err))

    if n_jobs <= 1 or len(payloads) <= 1:
        for payload in tqdm(payloads, total=len(payloads), desc=label, unit="song"):
            sid, err, duration = _mix_sum_check_job(payload)
            _handle(sid, err, duration)
            if _maybe_stop():
                break
    else:
        chunksize = max(1, min(16, len(payloads) // (n_jobs * 8) or 1))
        pbar = tqdm(total=len(payloads), desc=label, unit="song", miniters=1, smoothing=0.05)
        try:
            with multiprocessing.Pool(processes=n_jobs) as pool:
                for sid, err, duration in pool.imap_unordered(
                    _mix_sum_check_job, payloads, chunksize=chunksize,
                ):
                    pbar.update(1)
                    _handle(sid, err, duration)
                    if _maybe_stop():
                        pool.terminate()
                        break
        finally:
            pbar.close()
        bad.sort(key=lambda item: item[0])

    if persist_song_lengths and lengths:
        update_stems_csv_mix_column(root, song_lengths=lengths)
        print(
            f"Persisted song_length for {len(lengths)} song(s) on "
            f"{SPDMX_FILE_NAME}.csv (from decode+sum pass).",
            flush=True,
        )
    return bad


def verify_song_ids_match_stem_sums(
    media_dir: str | Path,
    song_ids: list[str],
    *,
    tables_dir: str | Path | None = None,
    jobs: int = 1,
    limit: int = 25,
) -> None:
    """Decode+sum-check only the given songs (dirty-mix post-check)."""
    del tables_dir  # reserved for claimed-stem path wiring; ids are explicit
    root = Path(media_dir)
    audio_root = root / SPDMX_AUDIO_DIR_NAME
    mix_root = root / SPDMX_MIX_DIR_NAME
    ids = [str(s) for s in song_ids]
    if not ids:
        return
    claimed = _claimed_stem_paths_by_song(root)
    payloads = [
        (
            sid,
            str(audio_root),
            str(mix_root),
            tuple(str(p) for p in claimed[sid]) if claimed and sid in claimed else None,
        )
        for sid in ids
    ]
    bad: list[tuple[str, str]] = []
    lengths: dict[str, float] = {}
    n_jobs = max(1, int(jobs))
    label = (
        f"verify touched mix=sum (-j {n_jobs})"
        if n_jobs > 1
        else "verify touched mix=sum"
    )
    if n_jobs <= 1 or len(payloads) <= 1:
        for payload in tqdm(payloads, total=len(payloads), desc=label, unit="song"):
            sid, err, duration = _mix_sum_check_job(payload)
            if duration is not None and duration > 0:
                lengths[sid] = float(duration)
            if err:
                bad.append((sid, err))
    else:
        chunksize = max(1, min(16, len(payloads) // (n_jobs * 8) or 1))
        with multiprocessing.Pool(processes=n_jobs) as pool:
            for sid, err, duration in tqdm(
                pool.imap_unordered(_mix_sum_check_job, payloads, chunksize=chunksize),
                total=len(payloads),
                desc=label,
                unit="song",
            ):
                if duration is not None and duration > 0:
                    lengths[sid] = float(duration)
                if err:
                    bad.append((sid, err))
        bad.sort(key=lambda item: item[0])

    if lengths:
        update_stems_csv_mix_column(root, song_lengths=lengths)

    if not bad:
        print(
            f"verify ok: {len(ids)} touched song(s) match sum(audio stems).",
            flush=True,
        )
        return

    n_bad = len(bad)
    show = bad[:limit]
    extra = f" (showing first {limit}; {n_bad} total)" if n_bad > limit else ""
    lines = "\n".join(f"  {msg}" for _sid, msg in show)
    raise RuntimeError(
        f"Touched mixes failed sample-wise sum check ({n_bad}{extra}):\n{lines}\n"
        "Re-run: uv run python -m synthesis.final --only-pass mix "
        "--repair-mix-sums -y -j 8"
    )


def verify_mixes_match_stem_sums(
    media_dir: str | Path,
    *,
    tables_dir: str | Path | None = None,
    jobs: int = 1,
    limit: int = 25,
    delete_bad: bool = False,
    force_delete: bool = False,
    also_delete_audio: bool = True,
) -> None:
    """Decode each song's stems+mix once and require mix == sample-wise stem sum.

    When ``delete_bad`` is True, mismatched mixes and (by default) their
    ``audio/<song_id>/`` trees are deleted so a later mix pass rebuilds from
    ``raw/``. ``raw/`` is never touched.

    Raises ``RuntimeError`` listing up to ``limit`` mismatches. Used by
    ``synthesis.final --only-pass verify`` (combined with FLAC decode).
    """
    song_ids = _song_ids_for_mix_sum_check(Path(media_dir), tables_dir=tables_dir)
    # When deleting, collect all mismatches (not just ``limit``) so cleanup is complete.
    bad = collect_mix_sum_mismatches(
        media_dir,
        tables_dir=tables_dir,
        jobs=jobs,
        limit=None if delete_bad else limit,
    )
    if not bad:
        print(
            f"verify ok: {len(song_ids)} song(s) decoded; "
            "mixes match sum(audio stems) sample-wise.",
            flush=True,
        )
        return

    n_bad = len(bad)
    show = bad[:limit]
    extra = (
        f" (showing first {limit}; {n_bad} total)"
        if n_bad > limit
        else ""
    )
    lines = "\n".join(f"  {msg}" for _sid, msg in show)

    if delete_bad:
        delete_mix_sum_mismatches(
            media_dir,
            tables_dir=tables_dir,
            jobs=jobs,
            mismatches=bad,
            force=force_delete,
            also_delete_audio=also_delete_audio,
        )
        raise RuntimeError(
            f"Mix files failed sample-wise sum check ({n_bad}{extra}); "
            f"deleted mix+audio so mix can remake from raw/:\n{lines}\n"
            "Re-run: uv run python -m synthesis.final --only-pass mix -j 8\n"
            "Then:    uv run python -m synthesis.final --only-pass verify -j 8"
        )

    raise RuntimeError(
        f"Mix files are not the sample-wise sum of audio stems ({n_bad}{extra}):\n"
        f"{lines}\n"
        "Delete:  uv run python -m synthesis.final --only-pass verify "
        "--delete-bad-mix-sums -y -j 8\n"
        "Then:    uv run python -m synthesis.final --only-pass mix -j 8\n"
        "  (or one-shot: --only-pass mix --repair-mix-sums -y -j 8)\n"
        "Then:    uv run python -m synthesis.final --only-pass verify -j 8"
    )


def delete_mix_sum_mismatches(
    media_dir: str | Path,
    *,
    tables_dir: str | Path | None = None,
    jobs: int = 8,
    mismatches: list[tuple[str, str]] | None = None,
    max_fraction: float = 0.05,
    max_count: int | None = None,
    force: bool = False,
    also_delete_audio: bool = True,
) -> list[tuple[str, str]]:
    """Delete ``mix/<song_id>.flac`` (and by default ``audio/<song_id>/``) on content sum failures.

    Only removes songs whose failure reason is safe (``max|mix-sum|``, length,
    or sample-rate mismatch). Never deletes on missing stems, read errors, or
    missing mixes (those would wipe good files during NFS/path glitches).

    By default also removes ``audio/<song_id>/`` so a later mix pass
    re-normalizes from ``raw/`` (needed when stems themselves are corrupt).
    ``raw/`` is never touched. Pass ``also_delete_audio=False`` only for
    mix-only debug remakes.

    Refuses to delete if the safe-to-delete count exceeds ``max_fraction`` of
    the catalog (default 5%) unless ``force`` is True — guards against a
    systematic false-positive wiping the tree.

    Returns the list of ``(song_id, reason)`` that were targeted for deletion.
    """
    import shutil

    root = Path(media_dir)
    mix_root = root / SPDMX_MIX_DIR_NAME
    audio_root = root / SPDMX_AUDIO_DIR_NAME
    all_ids = _song_ids_for_mix_sum_check(root, tables_dir=tables_dir)
    bad = (
        list(mismatches)
        if mismatches is not None
        else collect_mix_sum_mismatches(
            media_dir, tables_dir=tables_dir, jobs=jobs, limit=None,
        )
    )
    if not bad:
        print("Delete bad mixes: none (all match sum(audio stems)).", flush=True)
        return []

    deletable = [(sid, reason) for sid, reason in bad if mix_sum_failure_is_safe_to_delete(reason)]
    skipped = [(sid, reason) for sid, reason in bad if not mix_sum_failure_is_safe_to_delete(reason)]
    if skipped:
        print(
            f"Delete bad mixes: skipping {len(skipped)} non-content failure(s) "
            f"(missing stems / read errors / missing mix — not deleting):",
            flush=True,
        )
        for _sid, reason in skipped[:10]:
            print(f"  skip: {reason}", flush=True)
        if len(skipped) > 10:
            print(f"  … and {len(skipped) - 10} more", flush=True)

    if not deletable:
        print("Delete bad mixes: nothing safe to delete.", flush=True)
        return []

    n_catalog = max(len(all_ids), 1)
    frac = len(deletable) / n_catalog
    # Allow small absolute cleanups without --yes; block mass wipes on big trees.
    cap = max(100, int(max_fraction * n_catalog))
    if max_count is not None:
        cap = max_count
    if not force and len(deletable) > cap:
        raise RuntimeError(
            f"Refusing to delete {len(deletable)}/{n_catalog} mixes "
            f"({100 * frac:.1f}%; cap={cap} without --yes). "
            "This looks like a systematic false positive. "
            "Inspect a few reasons, then re-run with --yes if intentional.\n"
            + "\n".join(f"  {r}" for _s, r in deletable[:15])
        )

    what = "mix+audio" if also_delete_audio else "mix"
    print(
        f"Delete bad mixes: removing {len(deletable)}/{n_catalog} {what} "
        f"with content sum mismatches…",
        flush=True,
    )
    for sid, reason in deletable[:25]:
        print(f"  {reason}", flush=True)
    if len(deletable) > 25:
        print(f"  … and {len(deletable) - 25} more", flush=True)

    removed_mix = 0
    removed_audio = 0
    for sid, _reason in deletable:
        if ".." in Path(sid).parts or Path(sid).is_absolute():
            print(f"  warn: refuse unsafe song_id {sid!r}", flush=True)
            continue
        path = mix_root / f"{sid}.flac"
        try:
            resolved = path.resolve()
            mix_resolved = mix_root.resolve()
            if not str(resolved).startswith(str(mix_resolved) + "/"):
                print(f"  warn: refuse path outside mix/: {path}", flush=True)
                continue
            if resolved.is_file() and resolved.suffix.lower() == ".flac":
                resolved.unlink()
                removed_mix += 1
        except OSError as exc:
            print(f"  warn: could not delete {path}: {exc}", flush=True)

        if also_delete_audio:
            audio_dir = audio_root / sid
            try:
                audio_resolved = audio_dir.resolve()
                audio_root_resolved = audio_root.resolve()
                if not str(audio_resolved).startswith(str(audio_root_resolved) + "/"):
                    print(f"  warn: refuse path outside audio/: {audio_dir}", flush=True)
                    continue
                if audio_resolved.is_dir():
                    shutil.rmtree(audio_resolved)
                    removed_audio += 1
            except OSError as exc:
                print(f"  warn: could not delete {audio_dir}: {exc}", flush=True)

    if also_delete_audio:
        print(
            f"Delete bad mixes: removed mix={removed_mix} audio_dirs={removed_audio} "
            f"of {len(deletable)}; re-run mix to remake from raw/.",
            flush=True,
        )
    else:
        print(
            f"Delete bad mixes: removed {removed_mix}/{len(deletable)}; "
            "re-run mix to remake them.",
            flush=True,
        )
    return deletable


def repair_mix_sum_mismatches(
    media_dir: str | Path,
    *,
    tables_dir: str | Path | None = None,
    jobs: int = 8,
    force_delete: bool = False,
    also_delete_audio: bool = True,
) -> list[tuple[str, str]]:
    """Delete mixes (and by default ``audio/``) that fail the content sum check.

    Returns the deleted ``(song_id, reason)`` list. Prefer
    ``synthesis.final --only-pass mix --repair-mix-sums``, which then
    re-normalizes from ``raw/`` and remakes mixes.
    """
    return delete_mix_sum_mismatches(
        media_dir,
        tables_dir=tables_dir,
        jobs=jobs,
        force=force_delete,
        also_delete_audio=also_delete_audio,
    )


def _mix_ready(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def mix_needs_rerender(dest: Path, stem_paths: list[Path]) -> bool:
    """True when mix is missing/empty or any stem is newer than the mix file."""
    if not stem_paths:
        return False
    if not _mix_ready(dest):
        return True
    try:
        mix_mtime = dest.stat().st_mtime
    except OSError:
        return True
    for path in stem_paths:
        try:
            if path.stat().st_mtime > mix_mtime:
                return True
        except OSError:
            return True
    return False


def ffmpeg_sum_stems(
    stem_paths: list[Path],
    dest: Path,
    *,
    force: bool = False,
) -> str:
    """Write mono FLAC mix via ffmpeg ``amix`` (raw sum). Return status tag."""
    if not stem_paths:
        return "skip_no_stems"
    if not force and not mix_needs_rerender(dest, stem_paths):
        return "skip_exists"

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise FileNotFoundError("ffmpeg not found on PATH")

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    if tmp.exists():
        tmp.unlink()

    n = len(stem_paths)
    cmd: list[str] = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error"]
    for path in stem_paths:
        cmd.extend(["-i", str(path)])

    # Raw linear sum (normalize=0). Keep mono to match stem channel layout.
    inputs = "".join(f"[{i}:a]" for i in range(n))
    filter_complex = (
        f"{inputs}amix=inputs={n}:duration=longest:normalize=0,"
        f"aformat=channel_layouts=mono:sample_fmts=s16[aout]"
    )
    cmd.extend(
        [
            "-filter_complex",
            filter_complex,
            "-map",
            "[aout]",
            "-c:a",
            "flac",
            "-f",
            "flac",
            str(tmp),
        ]
    )
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not tmp.is_file() or tmp.stat().st_size <= 0:
        if tmp.exists():
            tmp.unlink()
        err = (proc.stderr or proc.stdout or "").strip() or f"exit {proc.returncode}"
        raise RuntimeError(f"ffmpeg failed for {dest}: {err}")
    tmp.replace(dest)
    return "wrote"


def song_ids_from_stems_csv(dataset_dir: Path) -> list[str]:
    csv_path = dataset_dir / TRACK_MAP_FILE_NAME
    if not csv_path.is_file():
        csv_path = dataset_dir / f"{SPDMX_FILE_NAME}.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Missing {dataset_dir / TRACK_MAP_FILE_NAME}; "
            "run prepare_synthesis / synthesis.final first."
        )
    table = pd.read_csv(csv_path, usecols=["song_id"])
    return sorted(table["song_id"].astype(str).unique())


def update_stems_csv_mix_column(
    dataset_dir: Path,
    *,
    song_lengths: dict[str, float] | None = None,
) -> None:
    """Set ``mix`` (and optionally ``song_length``) on the production track map."""
    csv_path = dataset_dir / f"{SPDMX_FILE_NAME}.csv"
    if not csv_path.is_file():
        return
    table = pd.read_csv(csv_path)
    if "song_id" not in table.columns:
        return
    table["mix"] = table["song_id"].astype(str).map(
        lambda sid: f"./{SPDMX_MIX_DIR_NAME}/{sid}.flac"
    )
    if song_lengths:
        lengths = {str(k): float(v) for k, v in song_lengths.items() if v is not None}
        if "song_length" not in table.columns:
            table["song_length"] = pd.NA
        mapped = table["song_id"].astype(str).map(lengths)
        table["song_length"] = mapped.where(mapped.notna(), table["song_length"])
    table.to_csv(csv_path, index=False)


def _mix_duration_seconds(path: Path) -> float | None:
    """Cheap FLAC header duration (no full decode)."""
    try:
        import soundfile as sf

        info = sf.info(str(path))
        if info.samplerate <= 0 or info.frames <= 0:
            return None
        return float(info.frames) / float(info.samplerate)
    except Exception:  # noqa: BLE001
        return None


def _render_one_song_mix(payload: dict) -> tuple[str, str, float | None]:
    """Picklable process-pool worker: render one song mix; return status + duration."""
    song_id = str(payload["song_id"])
    audio_root = Path(payload["audio_root"])
    mix_root = Path(payload["mix_root"])
    force = bool(payload["force"])
    dry_run = bool(payload["dry_run"])
    dest = mix_root / f"{song_id}.flac"
    stems = _stem_flac_paths(audio_root / song_id)
    if not stems:
        return "skip_no_stems", song_id, None
    if not force and not mix_needs_rerender(dest, stems):
        return "skip_exists", song_id, _mix_duration_seconds(dest)
    if dry_run:
        return "wrote", song_id, None
    try:
        status = ffmpeg_sum_stems(stems, dest, force=True)
        duration = _mix_duration_seconds(dest) if status == "wrote" else None
        return status, song_id, duration
    except Exception as exc:  # noqa: BLE001
        print(f"error: mix {song_id}: {exc}", file=sys.stderr)
        return "error", song_id, None


def render_dataset_mixes(
    dataset_dir: str | Path,
    *,
    jobs: int = 8,
    force: bool = False,
    dry_run: bool = False,
    update_csv: bool = True,
    force_ids: set[str] | None = None,
    song_ids: list[str] | None = None,
) -> dict[str, int]:
    """Render mixes for songs in the track map. Returns status counts.

    Dirty-aware: existing non-empty mixes that are at least as new as every
    audio stem are skipped inside each worker unless ``force`` is set or the
    song id is in ``force_ids``. Uses a ``multiprocessing.Pool`` when
    ``jobs > 1``. Writes ``mix`` + ``song_length`` on the track map when
    ``update_csv`` is True.
    """
    root = Path(dataset_dir)
    audio_root = root / SPDMX_AUDIO_DIR_NAME
    mix_root = root / SPDMX_MIX_DIR_NAME
    if not audio_root.is_dir():
        raise FileNotFoundError(f"Missing flat audio tree: {audio_root}")

    all_ids = song_ids if song_ids is not None else song_ids_from_stems_csv(root)
    forced = {str(s) for s in (force_ids or set())}
    counts = {"wrote": 0, "skip_exists": 0, "skip_no_stems": 0, "error": 0}

    payloads = [
        {
            "song_id": song_id,
            "audio_root": str(audio_root),
            "mix_root": str(mix_root),
            "force": bool(force or song_id in forced),
            "dry_run": dry_run,
        }
        for song_id in all_ids
    ]

    n_jobs = max(1, int(jobs))
    results: list[tuple[str, str, float | None]] = []
    if not payloads:
        pass
    elif n_jobs <= 1 or len(payloads) <= 1:
        results = [
            _render_one_song_mix(payload)
            for payload in tqdm(payloads, desc="render mixes", unit="song")
        ]
    else:
        label = f"render mixes (-j {n_jobs})"
        chunksize = max(1, min(8, len(payloads) // (n_jobs * 8) or 1))
        with multiprocessing.Pool(processes=n_jobs) as pool:
            for item in tqdm(
                pool.imap_unordered(_render_one_song_mix, payloads, chunksize=chunksize),
                total=len(payloads),
                desc=label,
                unit="song",
            ):
                results.append(item)

    lengths: dict[str, float] = {}
    for status, song_id, duration in results:
        counts[str(status)] = counts.get(str(status), 0) + 1
        if duration is not None and duration > 0:
            lengths[str(song_id)] = float(duration)

    if update_csv and not dry_run:
        update_stems_csv_mix_column(root, song_lengths=lengths or None)
    return counts


def main(argv=None) -> int:
    args = parse_args(argv)
    dataset_dir = Path(
        args.dataset_dir
        if args.dataset_dir is not None
        else spdmx_dev_dir(args.output_dir)
    )
    # Touch helpers so imports stay accurate for callers.
    _ = (spdmx_audio_dir(args.output_dir), spdmx_mix_dir(args.output_dir))
    try:
        if args.dry_run and (args.repair_sums or args.delete_bad_sums):
            bad = collect_mix_sum_mismatches(dataset_dir, jobs=args.jobs)
            action = "remade" if args.repair_sums else "deleted"
            print(f"dry-run: {len(bad)} mix(es) would be {action}")
            for _sid, reason in bad[:25]:
                print(f"  {reason}")
            return 0
        if args.repair_sums:
            # Debug CLI: remake mixes only. Full audio+mix repair is
            # ``synthesis.final --only-pass mix --repair-mix-sums``.
            bad = repair_mix_sum_mismatches(
                dataset_dir, jobs=args.jobs, also_delete_audio=False,
            )
            if not bad:
                counts = {
                    "wrote": 0, "skip_exists": 0, "skip_no_stems": 0, "error": 0,
                }
            else:
                counts = render_dataset_mixes(
                    dataset_dir,
                    jobs=args.jobs,
                    force_ids={sid for sid, _ in bad},
                )
        elif args.delete_bad_sums:
            delete_mix_sum_mismatches(dataset_dir, jobs=args.jobs)
            return 0
        else:
            counts = render_dataset_mixes(
                dataset_dir,
                jobs=args.jobs,
                force=args.force,
                dry_run=args.dry_run,
            )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    mode = "dry-run" if args.dry_run else "done"
    print(
        f"{mode}: {dataset_dir / SPDMX_MIX_DIR_NAME}/ "
        f"wrote={counts.get('wrote', 0)} "
        f"skip_exists={counts.get('skip_exists', 0)} "
        f"skip_no_stems={counts.get('skip_no_stems', 0)} "
        f"error={counts.get('error', 0)}"
    )
    return 1 if counts.get("error", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
