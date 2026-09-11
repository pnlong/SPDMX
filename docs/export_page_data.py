"""Export project-page JSON, chunk-layout figure, and short audio demos.

Reads the packaged release under ``SPDMX_OUTPUT_DIR/SPDMX`` (override with
``--spdmx-root``). Writes into ``docs/``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from analysis.gm_programs import DRUM_GM_ID, GM_PROGRAM_NAMES
from analysis.plots import plot_chunk_layout
from shared.config import OUTPUT_DIR
from synthesis.chunking import DEFAULT_NUM_CHUNKS
from synthesis.patches import _gm_class

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = Path(__file__).resolve().parent
DEFAULT_SPDMX = Path(OUTPUT_DIR) / "SPDMX"
DEFAULT_DEV = Path(OUTPUT_DIR) / "SPDMX_dev"
STATS_PATH = REPO_ROOT / "submission" / "data" / "dataset_stats.json"
SONG_LENGTH_REPORT = (
    REPO_ROOT / "analysis" / "output" / "song_lengths" / "song_length_report.json"
)
DEFAULT_STEM_RECIPE = Path(OUTPUT_DIR) / "dev" / "final" / "stem_recipe.csv"

CLIP_SECONDS = 18
CLIP_HOP_SECONDS = 1.0
MAX_SCAN_SECONDS = 90.0
MIN_STEM_RMS = 0.008
CANDIDATES_PER_DEMO = 48
# Prefer SPDMX_dev for mixes (current on-disk chunk layout may still be split).
DEMO_SPECS: list[dict] = [
    {
        "id": "solo_piano",
        "label": "Solo piano",
        "blurb": "Single-stem Acoustic Grand — typical of ~65% of the corpus.",
        "filter": "solo_piano",
        "require_all_active": False,
    },
    {
        "id": "bdgp",
        "label": "Bass · Drums · Guitar · Piano",
        "blurb": "Four-stem BDGP subset matching common separation targets (bass, drums, guitar, piano).",
        "filter": "bdgp",
        "require_all_active": True,
    },
    {
        "id": "ensemble",
        "label": "Choir / ensemble",
        "blurb": "Multi-stem writing in the GM ensemble family (e.g. choir aahs).",
        "filter": "ensemble",
        "require_all_active": True,
    },
    {
        "id": "orchestral",
        "label": "Orchestral hybrid",
        "blurb": "Mixed orchestral families — strings, brass, reeds, and more.",
        "filter": "orchestral",
        "require_all_active": True,
    },
]


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _histogram(values: pd.Series, *, max_bin: int = 20, min_bin: int = 1) -> dict:
    counts = Counter(int(v) for v in values.dropna().astype(int))
    labels: list[str] = []
    data: list[int] = []
    for n in range(min_bin, max_bin + 1):
        labels.append(str(n))
        data.append(int(counts.get(n, 0)))
    overflow = sum(c for k, c in counts.items() if k > max_bin)
    if overflow:
        labels.append(f">{max_bin}+")
        data.append(int(overflow))
    return {"labels": labels, "counts": data}


def export_tracks_per_song(songs: pd.DataFrame, *, max_bin: int = 20) -> dict:
    """Histogram of stems/song for multi-stem songs only, plus single-stem caption stats."""
    n_tracks = songs["n_tracks"].dropna().astype(int)
    n_total = int(len(n_tracks))
    n_single = int((n_tracks == 1).sum())
    multi = n_tracks[n_tracks >= 2]
    hist = _histogram(multi, max_bin=max_bin, min_bin=2)
    hist.update(
        {
            "n_total": n_total,
            "n_single": n_single,
            "frac_single": (n_single / n_total) if n_total else 0.0,
            "n_multi": int(len(multi)),
            "note": "Bars exclude single-stem songs.",
        }
    )
    return hist


def _program_name(program: int, is_drum: bool) -> str:
    if is_drum or program == DRUM_GM_ID:
        return "Drums"
    if 0 <= program < len(GM_PROGRAM_NAMES):
        return GM_PROGRAM_NAMES[program]
    return f"Program {program}"


def release_hours_from_songs(songs: pd.DataFrame) -> float | None:
    """Mixture hours: sum of per-song mix lengths / 3600 (never multiply by stem count)."""
    if "song_length" not in songs.columns:
        return None
    lengths = pd.to_numeric(songs["song_length"], errors="coerce").dropna()
    lengths = lengths[lengths > 0]
    if lengths.empty:
        return None
    return float(lengths.sum() / 3600.0)


def export_summary(stats: dict, songs: pd.DataFrame, chunks: pd.DataFrame) -> dict:
    hours = release_hours_from_songs(songs)
    if hours is None:
        hours = float(stats.get("release_hours_approx", 6248))
    return {
        "release_songs": int(stats.get("release_songs", len(songs))),
        "release_stems": int(stats.get("release_stems", 0)),
        # One mix duration per song (song_length); do not sum stem lengths.
        "release_hours": round(hours, 1),
        "release_hours_approx": round(hours, 1),
        "hours_source": (
            "sum(songs.csv song_length) / 3600 — mix duration per song, not stems"
            if "song_length" in songs.columns
            else "dataset_stats fallback"
        ),
        "sample_rate_hz": stats.get("sample_rate_hz", 44100),
        "channels": stats.get("channels", 1),
        "audio_format": stats.get("audio_format", "flac"),
        "target_lufs": stats.get("target_lufs", -23.0),
        "n_chunks": int(len(chunks)),
        "chunk_target_gib": (
            float(chunks["bytes"].mean()) / (1024**3) if len(chunks) else None
        ),
        "default_num_chunks": DEFAULT_NUM_CHUNKS,
        "mean_stems_per_song": float(stats.get("mean_stems_per_song", songs["n_tracks"].mean())),
        "median_stems_per_song": float(
            stats.get("median_stems_per_song", songs["n_tracks"].median())
        ),
        "max_stems_per_song": int(stats.get("max_stems_per_song", songs["n_tracks"].max())),
        "songs_with_gt1_stem_frac": float(
            stats.get(
                "songs_with_gt1_stem_frac",
                (songs["n_tracks"] > 1).mean(),
            )
        ),
        "subset_bdgp_songs": int(
            stats.get("subset_bdgp_songs", songs["subset:bdgp"].sum())
        ),
        "backends": stats.get("backends", {}),
    }


def export_gm_classes(songs: pd.DataFrame) -> dict:
    counter: Counter[str] = Counter()
    for raw in songs["gm_classes"].fillna(""):
        for part in str(raw).split("|"):
            key = part.strip()
            if key:
                counter[key] += 1
    items = counter.most_common()
    return {
        "labels": [k for k, _ in items],
        "counts": [int(v) for _, v in items],
        "note": "Song-level: a song contributes once per distinct GM class present.",
    }


def export_programs_top(stems: pd.DataFrame, *, top_n: int = 20) -> dict:
    work = stems.copy()
    work["label"] = [
        _program_name(int(p), bool(d))
        for p, d in zip(work["program"], work["is_drum"], strict=False)
    ]
    counts = work["label"].value_counts().head(top_n)
    return {"labels": counts.index.tolist(), "counts": [int(v) for v in counts.values]}


def export_chunks(chunks: pd.DataFrame) -> dict:
    df = chunks.sort_values("chunk")
    mean_gib = float(df["bytes"].mean()) / (1024**3) if len(df) else 0.0
    return {
        "chunk": [int(c) for c in df["chunk"]],
        "n_songs": [int(v) for v in df["n_songs"]],
        "n_stems": [int(v) for v in df["n_stems"]],
        "bytes": [int(v) for v in df["bytes"]],
        "gib": [round(int(v) / (1024**3), 3) for v in df["bytes"]],
        "target_gib": mean_gib,
        "num_chunks": int(len(df)) if len(df) else DEFAULT_NUM_CHUNKS,
    }


def _candidate_songs(songs: pd.DataFrame, kind: str) -> pd.DataFrame:
    s = songs
    if kind == "solo_piano":
        hit = s[
            (s["n_tracks"] == 1)
            & s["gm_classes"].fillna("").str.fullmatch("piano", case=False)
        ]
    elif kind == "bdgp":
        hit = s[s["subset:bdgp"] & (s["n_tracks"].between(4, 8))]
    elif kind == "ensemble":
        hit = s[
            s["gm_classes"].fillna("").str.contains(r"(?:^|\|)ensemble(?:\||$)", regex=True)
            & (s["n_tracks"].between(3, 8))
        ]
    elif kind == "orchestral":
        hit = s[
            s["gm_classes"].fillna("").str.contains("string|brass|reed", case=False, regex=True)
            & (s["n_tracks"].between(4, 10))
            & ~s["subset:bdgp"]
        ]
    else:
        return s.iloc[0:0]
    if hit.empty:
        return hit
    # Spread candidates across the corpus; prefer moderate track counts.
    return hit.sort_values(["n_tracks", "song_id"]).reset_index(drop=True)


def _load_mono(path: Path, *, max_seconds: float = MAX_SCAN_SECONDS) -> tuple[np.ndarray, int]:
    info = sf.info(str(path))
    sr = int(info.samplerate)
    frames = min(int(info.frames), int(sr * max_seconds))
    y, _ = sf.read(str(path), dtype="float32", always_2d=True, frames=frames)
    return y.mean(axis=1), sr


def _window_rms(y: np.ndarray, sr: int, start_s: float, dur_s: float) -> float:
    a = int(start_s * sr)
    b = min(len(y), int((start_s + dur_s) * sr))
    if b <= a:
        return 0.0
    seg = y[a:b]
    return float(np.sqrt(np.mean(seg * seg) + 1e-12))


def _best_all_active_start(stem_paths: list[Path]) -> tuple[float, float] | None:
    """Return (start_sec, score) where every stem is active in the clip window."""
    loaded: list[tuple[np.ndarray, int]] = []
    for path in stem_paths:
        try:
            loaded.append(_load_mono(path))
        except Exception:
            return None
    if not loaded:
        return None

    min_dur = min(len(y) / sr for y, sr in loaded)
    if min_dur < CLIP_SECONDS + 1.0:
        return None

    best: tuple[float, float] | None = None
    t = 0.0
    while t + CLIP_SECONDS <= min_dur:
        rms = [_window_rms(y, sr, t, CLIP_SECONDS) for y, sr in loaded]
        if min(rms) >= MIN_STEM_RMS:
            # Prefer balanced activity (high min, not just one loud stem).
            score = float(min(rms) * (1.0 + float(np.mean(rms))))
            if best is None or score > best[1]:
                best = (t, score)
        t += CLIP_HOP_SECONDS
    return best


def _stem_paths_for_song(
    song_id: str,
    *,
    spdmx_root: Path,
    dev_root: Path,
    chunk: int,
    n_tracks: int,
) -> tuple[list[Path], Path | None]:
    """Resolve stem FLACs + mix; prefer flat release, fall back to SPDMX_dev."""
    flat_dir = spdmx_root / f"chunk_{chunk}" / song_id
    split_dir = spdmx_root / f"chunk_{chunk}" / "audio" / song_id
    dev_dir = dev_root / "audio" / song_id

    stem_dir = next((d for d in (flat_dir, split_dir, dev_dir) if d.is_dir()), None)
    stems: list[Path] = []
    if stem_dir is not None:
        for t in range(int(n_tracks)):
            p = stem_dir / f"{t}.flac"
            if p.is_file():
                stems.append(p)

    mix_candidates = [
        flat_dir / "mix.flac",
        spdmx_root / f"chunk_{chunk}" / "mix" / f"{song_id}.flac",
        dev_root / "mix" / f"{song_id}.flac",
    ]
    mix = next((p for p in mix_candidates if p.is_file()), None)
    return stems, mix


def _ffmpeg_clip(
    src: Path,
    dst: Path,
    *,
    start_s: float = 0.0,
    seconds: float = CLIP_SECONDS,
) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start_s:.3f}",
        "-t",
        str(seconds),
        "-i",
        str(src),
        "-codec:a",
        "libmp3lame",
        "-q:a",
        "4",
        str(dst),
    ]
    try:
        subprocess.run(cmd, check=True)
        return dst.is_file() and dst.stat().st_size > 0
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _select_demo_song(
    songs: pd.DataFrame,
    *,
    kind: str,
    require_all_active: bool,
    spdmx_root: Path,
    dev_root: Path,
) -> tuple[pd.Series, list[Path], Path | None, float] | None:
    candidates = _candidate_songs(songs, kind)
    if candidates.empty:
        return None

    # Sample evenly across the candidate list.
    n = len(candidates)
    idxs = np.unique(
        np.linspace(0, n - 1, num=min(CANDIDATES_PER_DEMO, n), dtype=int)
    ).tolist()

    fallback: tuple[pd.Series, list[Path], Path | None, float, float] | None = None
    for i in idxs:
        row = candidates.iloc[int(i)]
        song_id = str(row["song_id"])
        chunk = int(row["chunk"])
        n_tracks = int(row["n_tracks"])
        stem_paths, mix_path = _stem_paths_for_song(
            song_id,
            spdmx_root=spdmx_root,
            dev_root=dev_root,
            chunk=chunk,
            n_tracks=n_tracks,
        )
        if len(stem_paths) < max(1, n_tracks):
            continue
        if not require_all_active:
            return row, stem_paths, mix_path, 5.0

        hit = _best_all_active_start(stem_paths)
        if hit is None:
            continue
        start_s, score = hit
        if fallback is None or score > fallback[4]:
            fallback = (row, stem_paths, mix_path, start_s, score)
        # Good enough: stop early once we have a strong all-active window.
        if score >= 0.03:
            return row, stem_paths, mix_path, start_s

    if fallback is None:
        return None
    row, stem_paths, mix_path, start_s, _score = fallback
    return row, stem_paths, mix_path, start_s


def export_audio_demos(
    songs: pd.DataFrame,
    stems: pd.DataFrame,
    *,
    spdmx_root: Path,
    dev_root: Path,
    audio_dir: Path,
) -> dict:
    audio_dir.mkdir(parents=True, exist_ok=True)
    demos: list[dict] = []
    for spec in DEMO_SPECS:
        selected = _select_demo_song(
            songs,
            kind=spec["filter"],
            require_all_active=bool(spec.get("require_all_active")),
            spdmx_root=spdmx_root,
            dev_root=dev_root,
        )
        if selected is None:
            print(f"skip demo {spec['id']}: no matching song")
            continue
        row, stem_paths, mix_path, start_s = selected
        song_id = str(row["song_id"])
        chunk = int(row["chunk"])
        n_tracks = int(row["n_tracks"])

        stem_meta = stems[stems["song_id"] == song_id].sort_values("track")
        demo_dir = audio_dir / spec["id"]
        if demo_dir.exists():
            shutil.rmtree(demo_dir)
        demo_dir.mkdir(parents=True)

        stem_entries: list[dict] = []
        for _, srow in stem_meta.iterrows():
            track = int(srow["track"])
            src = next((p for p in stem_paths if p.stem == str(track)), None)
            if src is None:
                continue
            out = demo_dir / f"{track}.mp3"
            if not _ffmpeg_clip(src, out, start_s=start_s):
                continue
            prog = int(srow["program"])
            is_drum = bool(srow["is_drum"])
            stem_entries.append(
                {
                    "track": track,
                    "file": f"audio/{spec['id']}/{track}.mp3",
                    "label": _program_name(prog, is_drum),
                    "gm_class": _gm_class(prog, is_drum),
                    "program": prog,
                    "is_drum": is_drum,
                }
            )

        mix_file = None
        if mix_path is not None:
            mix_out = demo_dir / "mix.mp3"
            if _ffmpeg_clip(mix_path, mix_out, start_s=start_s):
                mix_file = f"audio/{spec['id']}/mix.mp3"

        if not stem_entries and not mix_file:
            print(f"skip demo {spec['id']}: ffmpeg failed")
            continue

        demos.append(
            {
                "id": spec["id"],
                "label": spec["label"],
                "blurb": spec["blurb"],
                "song_id": song_id,
                "chunk": chunk,
                "n_tracks": n_tracks,
                "clip_start_s": round(float(start_s), 2),
                "gm_classes": str(row.get("gm_classes", "")),
                "mix": mix_file,
                "stems": stem_entries,
            }
        )
        print(
            f"demo {spec['id']}: {song_id} "
            f"({len(stem_entries)} stems, start={start_s:.1f}s)"
        )

    return {"clip_seconds": CLIP_SECONDS, "demos": demos}


def export_render_backends(recipe_path: Path) -> dict:
    """Break production stems into Basic / Varied soundfonts / MIDI-DDSP.

    Uses ``stem_recipe.csv``: neural audio is ``backend=midi_ddsp``; soundfont
    audio is labeled by ``method`` (or ``fallback`` when a DDSP recipe fell
    back to FluidSynth). ``slakh`` in the recipe is the varied-soundfont path.
    """
    if not recipe_path.is_file():
        return {"labels": [], "counts": [], "note": f"missing {recipe_path}"}

    df = pd.read_csv(recipe_path, usecols=["method", "fallback", "backend"])
    labels_order = ["basic", "slakh", "midi_ddsp"]
    display = {
        "basic": "Basic (FluidSynth)",
        "slakh": "Varied soundfonts",
        "midi_ddsp": "MIDI-DDSP",
    }
    counts = {k: 0 for k in labels_order}

    for method, fallback, backend in zip(
        df["method"].astype(str),
        df["fallback"].astype(str),
        df["backend"].astype(str),
        strict=False,
    ):
        backend_l = backend.strip().lower()
        method_l = method.strip().lower().replace("-", "_")
        fallback_l = fallback.strip().lower().replace("-", "_")
        if backend_l == "midi_ddsp":
            counts["midi_ddsp"] += 1
        elif method_l in ("basic", "slakh"):
            counts[method_l] += 1
        elif fallback_l in ("basic", "slakh"):
            counts[fallback_l] += 1
        else:
            # Unknown soundfont path — fold into slakh (dominant production default).
            counts["slakh"] += 1

    return {
        "labels": [display[k] for k in labels_order],
        "ids": labels_order,
        "counts": [int(counts[k]) for k in labels_order],
        "note": "Soundfont stems: basic (single bank) vs varied (per-category pools); MIDI-DDSP is neural.",
    }


# Uneven bins emphasize the short-song mass and SA3-relevant cutoffs (120s / 380s).
DURATION_HIST_EDGES_S = (
    0,
    15,
    30,
    45,
    60,
    90,
    120,
    180,
    240,
    300,
    380,
    600,
    1200,
    float("inf"),
)
DURATION_HIST_LABELS = (
    "0–15",
    "15–30",
    "30–45",
    "45–60",
    "60–90",
    "90–120",
    "120–180",
    "180–240",
    "240–300",
    "300–380",
    "380–600",
    "600–1200",
    "1200+",
)


def export_duration_from_songs(songs: pd.DataFrame) -> dict | None:
    """Histogram + percentiles of mix ``song_length`` (soundfile seconds)."""
    if "song_length" not in songs.columns:
        return None
    lengths = pd.to_numeric(songs["song_length"], errors="coerce").dropna()
    lengths = lengths[lengths > 0]
    if lengths.empty:
        return None

    edges = list(DURATION_HIST_EDGES_S)
    cats = pd.cut(
        lengths,
        bins=edges,
        labels=list(DURATION_HIST_LABELS),
        right=False,
        include_lowest=True,
    )
    counts = [int((cats == lab).sum()) for lab in DURATION_HIST_LABELS]
    n = int(len(lengths))
    p50 = float(lengths.median())
    p95 = float(lengths.quantile(0.95))
    p99 = float(lengths.quantile(0.99))
    pct_120 = float((lengths <= 120).mean() * 100.0)
    pct_380 = float((lengths <= 380).mean() * 100.0)
    hours = float(lengths.sum() / 3600.0)
    return {
        "source": "songs.csv song_length (mix soundfile duration)",
        "unit": "seconds",
        "labels": list(DURATION_HIST_LABELS),
        "counts": counts,
        "bin_edges_seconds": [None if e == float("inf") else e for e in edges],
        "n_songs": n,
        "hours": round(hours, 1),
        "percentiles": {
            "p50": round(p50, 1),
            "p75": round(float(lengths.quantile(0.75)), 1),
            "p90": round(float(lengths.quantile(0.90)), 1),
            "p95": round(p95, 1),
            "p99": round(p99, 1),
        },
        "summary": {
            "median_song_duration_seconds": round(p50, 1),
            "mean_song_duration_seconds": round(float(lengths.mean()), 1),
            "pct_songs_under_120s": round(pct_120, 1),
            "pct_songs_under_380s": round(pct_380, 1),
            "n_songs_over_380s": int((lengths > 380).sum()),
        },
        "caption": (
            f"Mix durations peak under a minute (median {p50:.0f} s); "
            f"{pct_120:.0f}% of songs are ≤120 s, with a long tail of multi-minute scores "
            f"({int((lengths > 380).sum()):,} over 380 s)."
        ),
    }


def export_duration(report_path: Path) -> dict | None:
    """Legacy PDMX-metadata duration report (fallback if songs lack song_length)."""
    if not report_path.is_file():
        return None
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return {
        "source": "PDMX song_length.seconds (legacy)",
        "percentiles": report.get("percentiles", {}),
        "summary": report.get("summary", {}),
        "sa3_limits": report.get("sa3_limits", {}),
    }


def _soundfont_display_name(entry: dict) -> str:
    raw = entry.get("archive_name") or entry.get("file") or entry.get("id") or ""
    return Path(str(raw)).name


def export_soundfonts() -> dict:
    """Unique FluidSynth banks from the locked varied shortlists (+ basic default)."""
    import yaml

    from experiments.patch_sweep.config import (
        WINNERS_LOCKED_PATH,
        load_combined_soundfont_catalog,
    )
    from shared.config import SOUNDFONT_PATH

    catalog = {
        c["id"]: c for c in load_combined_soundfont_catalog().get("candidates", [])
    }
    locked = yaml.safe_load(WINNERS_LOCKED_PATH.read_text(encoding="utf-8")) or {}
    by_category: dict[str, list[dict]] = {}
    unique: dict[str, dict] = {}
    for category, cfg in (locked.get("categories") or {}).items():
        rows: list[dict] = []
        for sid in cfg.get("soundfont_ids") or []:
            entry = catalog.get(sid, {"id": sid, "file": sid})
            name = _soundfont_display_name(entry)
            row = {"id": sid, "file": name}
            rows.append(row)
            unique[sid] = row
        by_category[str(category)] = rows

    fonts = sorted(unique.values(), key=lambda r: r["file"].lower())
    basic_file = Path(SOUNDFONT_PATH).name
    archive_url = "https://archive.org/download/free-soundfonts-sf2-2019-04"
    return {
        "basic": {
            "name": "SGM-V2.01",
            "file": basic_file,
            "note": "Single-bank FluidSynth path for basic / fallback stems.",
        },
        "source": "experiments/patch_sweep/winners_locked.yaml",
        "collection": "Archive.org free-soundfonts-sf2-2019-04",
        "collection_url": archive_url,
        "n_varied": len(fonts),
        "fonts": fonts,
        "by_category": by_category,
        "caption": (
            "Varied renders sample a listening-selected shortlist per instrument "
            "category; basic renders use SGM-V2.01."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spdmx-root", type=Path, default=DEFAULT_SPDMX)
    parser.add_argument("--dev-root", type=Path, default=DEFAULT_DEV)
    parser.add_argument("--docs-dir", type=Path, default=DOCS_DIR)
    parser.add_argument("--stem-recipe", type=Path, default=DEFAULT_STEM_RECIPE)
    parser.add_argument("--skip-audio", action="store_true")
    parser.add_argument("--skip-figure", action="store_true")
    args = parser.parse_args()

    root: Path = args.spdmx_root
    docs: Path = args.docs_dir
    data_dir = docs / "data"
    assets_dir = docs / "assets"
    audio_dir = docs / "audio"
    data_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    songs = pd.read_csv(root / "songs.csv")
    stems = pd.read_csv(root / "stems.csv")
    chunks = pd.read_csv(root / "chunks.csv")
    stats = (
        json.loads(STATS_PATH.read_text(encoding="utf-8"))
        if STATS_PATH.is_file()
        else {}
    )

    summary = export_summary(stats, songs, chunks)
    _write_json(data_dir / "summary.json", summary)
    _write_json(data_dir / "tracks_per_song.json", export_tracks_per_song(songs))
    _write_json(data_dir / "gm_classes.json", export_gm_classes(songs))
    _write_json(data_dir / "programs_top.json", export_programs_top(stems))
    _write_json(data_dir / "chunks.json", export_chunks(chunks))
    comparison = {
        k: dict(v)
        for k, v in stats.get("comparison_table", {}).items()
        if str(k).upper() != "PDMX"
        and str((v or {}).get("type", "")).lower() != "symbolic"
    }
    # Peer chart: SPDMX hours from mix song_length, not PDMX symbolic duration.
    if "SPDMX" in comparison and summary.get("release_hours") is not None:
        comparison["SPDMX"]["hours"] = summary["release_hours"]
        comparison["SPDMX"]["songs"] = summary.get(
            "release_songs", comparison["SPDMX"].get("songs")
        )
    _write_json(data_dir / "comparison.json", comparison)
    backends = export_render_backends(args.stem_recipe)
    _write_json(data_dir / "backends.json", backends)
    # Keep summary.backends aligned with the page chart.
    if backends.get("ids") and backends.get("counts"):
        summary["backends"] = {
            i: c for i, c in zip(backends["ids"], backends["counts"], strict=False)
        }
        _write_json(data_dir / "summary.json", summary)
    duration = export_duration_from_songs(songs) or export_duration(SONG_LENGTH_REPORT)
    if duration:
        _write_json(data_dir / "duration.json", duration)
    _write_json(data_dir / "soundfonts.json", export_soundfonts())

    if not args.skip_figure:
        for suffix in (".pdf", ".png", ".svg"):
            out = assets_dir / f"chunk_layout{suffix}"
            plot_chunk_layout(chunks, out)
            # also paper figures PDF
            if suffix == ".pdf":
                paper = REPO_ROOT / "submission" / "figs" / "chunk_layout.pdf"
                plot_chunk_layout(chunks, paper)
            print(f"Wrote {out}")

    if not args.skip_audio:
        manifest = export_audio_demos(
            songs,
            stems,
            spdmx_root=root,
            dev_root=args.dev_root,
            audio_dir=audio_dir,
        )
        _write_json(audio_dir / "manifest.json", manifest)
    else:
        print("skip audio demos")

    print(f"Done → {docs}")


if __name__ == "__main__":
    main()
