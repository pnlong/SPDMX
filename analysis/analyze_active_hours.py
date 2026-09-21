"""Compute RMS-gated active hours per GM program from SPDMX stems.

Reads ``stems.csv`` under a flat ``SPDMX_dev`` (or chunked release) root,
measures each stem's wall-clock and short-hop RMS-active duration, and writes
per-stem + per-program tables plus a preview plot.

Full-corpus scans are slow (FLAC decode bound). Use ``--limit`` / ``--sample``
for a quick look; omit them for the full release.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from analysis.active_hours import (
    DEFAULT_HOP_SECONDS,
    DEFAULT_MIN_RMS,
    aggregate_hours_by_program,
    gm_id_from_stem_row,
    measure_stem_active_seconds,
)
from analysis.plots import plot_instrument_active_hours
from shared.config import OUTPUT_DIR, SPDMX_DEV_DIR_NAME, SPDMX_FILE_NAME
from synthesis.paths import analysis_root

_WORKER_ROOT: Path | None = None
_WORKER_HOP = DEFAULT_HOP_SECONDS
_WORKER_MIN_RMS = DEFAULT_MIN_RMS


def _default_spdmx_dev() -> Path:
    share = Path("/deepfreeze/share/SPDMX") / SPDMX_DEV_DIR_NAME
    if (share / f"{SPDMX_FILE_NAME}.csv").is_file():
        return share
    return Path(OUTPUT_DIR) / SPDMX_DEV_DIR_NAME


def _default_output_dir() -> Path:
    return Path(analysis_root(OUTPUT_DIR)) / "active_hours"


def parse_args(args=None, namespace=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spdmx-root",
        type=Path,
        default=_default_spdmx_dev(),
        help="Flat SPDMX_dev root containing stems.csv and audio/.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: {OUTPUT_DIR}/dev/analysis/active_hours).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Use only the first N stem rows.")
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Random sample of N stems (after --limit, if any).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("-j", "--jobs", type=int, default=max(1, mp.cpu_count() // 2))
    parser.add_argument("--hop-seconds", type=float, default=DEFAULT_HOP_SECONDS)
    parser.add_argument("--min-rms", type=float, default=DEFAULT_MIN_RMS)
    parser.add_argument("--top-n", type=int, default=15)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip stems already present in stem_active_seconds.csv.",
    )
    return parser.parse_args(args=args, namespace=namespace)


def _init_worker(root: str, hop_seconds: float, min_rms: float) -> None:
    global _WORKER_ROOT, _WORKER_HOP, _WORKER_MIN_RMS
    _WORKER_ROOT = Path(root)
    _WORKER_HOP = float(hop_seconds)
    _WORKER_MIN_RMS = float(min_rms)


def _measure_row(payload: tuple[str, int, int, object, object]) -> dict:
    song_id, track, gm_id, program, is_drum = payload
    assert _WORKER_ROOT is not None
    flac = _WORKER_ROOT / "audio" / song_id / f"{int(track)}.flac"
    if not flac.is_file():
        return {
            "song_id": song_id,
            "track": int(track),
            "gm_id": int(gm_id),
            "program": program,
            "is_drum": bool(is_drum),
            "wall_seconds": float("nan"),
            "active_seconds": float("nan"),
            "ok": False,
        }
    measured = measure_stem_active_seconds(
        flac,
        hop_seconds=_WORKER_HOP,
        min_rms=_WORKER_MIN_RMS,
    )
    if measured is None:
        wall = active = float("nan")
        ok = False
    else:
        wall, active = measured
        ok = True
    return {
        "song_id": song_id,
        "track": int(track),
        "gm_id": int(gm_id),
        "program": program,
        "is_drum": bool(is_drum),
        "wall_seconds": wall,
        "active_seconds": active,
        "ok": ok,
    }


def _stem_payloads(stems: pd.DataFrame) -> list[tuple[str, int, int, object, object]]:
    payloads: list[tuple[str, int, int, object, object]] = []
    for row in stems.itertuples(index=False):
        song_id = str(getattr(row, "song_id"))
        track = int(getattr(row, "track"))
        program = getattr(row, "program")
        is_drum = getattr(row, "is_drum")
        gm_id = gm_id_from_stem_row(program, is_drum)
        payloads.append((song_id, track, gm_id, program, is_drum))
    return payloads


def main(args=None) -> None:
    opts = parse_args(args)
    root = opts.spdmx_root
    stems_csv = root / f"{SPDMX_FILE_NAME}.csv"
    if not stems_csv.is_file():
        raise SystemExit(f"missing stems table: {stems_csv}")

    out_dir = opts.output_dir or _default_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem_out = out_dir / "stem_active_seconds.csv"
    program_out = out_dir / "program_active_hours.csv"
    fig_out = out_dir / "instrument_active_hours.pdf"

    stems = pd.read_csv(stems_csv)
    if opts.limit is not None:
        stems = stems.head(int(opts.limit))
    if opts.sample is not None:
        n = min(int(opts.sample), len(stems))
        stems = stems.sample(n=n, random_state=int(opts.seed))

    done_keys: set[tuple[str, int]] = set()
    prior: pd.DataFrame | None = None
    if opts.resume and stem_out.is_file():
        prior = pd.read_csv(stem_out)
        done = prior.loc[prior["ok"] == True, ["song_id", "track"]].copy()  # noqa: E712
        done["song_id"] = done["song_id"].astype(str)
        done["track"] = done["track"].astype(int)
        done_keys = set(zip(done["song_id"], done["track"]))
        before = len(stems)
        mask = [
            (str(sid), int(tr)) not in done_keys
            for sid, tr in zip(stems["song_id"], stems["track"])
        ]
        stems = stems.loc[mask]
        print(f"resume: skipping {before - len(stems)} finished stems", flush=True)

    payloads = _stem_payloads(stems)
    print(
        f"measuring {len(payloads)} stems under {root} "
        f"(jobs={opts.jobs}, hop={opts.hop_seconds}s, min_rms={opts.min_rms})",
        flush=True,
    )

    records: list[dict] = []
    if opts.jobs <= 1:
        _init_worker(str(root), opts.hop_seconds, opts.min_rms)
        for payload in tqdm(payloads, desc="active-hours", unit="stem"):
            records.append(_measure_row(payload))
    else:
        with mp.Pool(
            processes=int(opts.jobs),
            initializer=_init_worker,
            initargs=(str(root), opts.hop_seconds, opts.min_rms),
        ) as pool:
            for row in tqdm(
                pool.imap(_measure_row, payloads, chunksize=32),
                total=len(payloads),
                desc="active-hours",
                unit="stem",
            ):
                records.append(row)

    measured = pd.DataFrame.from_records(records)
    if prior is not None and not prior.empty:
        measured = pd.concat([prior, measured], ignore_index=True)
        measured = measured.drop_duplicates(subset=["song_id", "track"], keep="last")

    measured.to_csv(stem_out, index=False)
    ok = measured[measured["ok"] == True]  # noqa: E712
    summary = aggregate_hours_by_program(ok)
    summary.to_csv(program_out, index=False)

    plot_instrument_active_hours(summary, fig_out, top_n=opts.top_n)

    # Keep paper data + regenerate Stems|Hours GM figure with RMS-active hours.
    paper_data = Path(__file__).resolve().parent / "paper_data"
    paper_data.mkdir(parents=True, exist_ok=True)
    (paper_data / "program_active_hours.csv").write_text(program_out.read_text())
    try:
        from analysis.make_figures import make_gm_program_compare_figure

        gm_fig = make_gm_program_compare_figure(top_n=None, rank_by="stems")
        print(f"Wrote {gm_fig}")
    except Exception as exc:  # pragma: no cover - best-effort paper refresh
        print(f"skip paper GM figure refresh: {exc}", flush=True)

    n_ok = int(len(ok))
    n_bad = int((measured["ok"] == False).sum())  # noqa: E712
    print(f"Wrote {stem_out} ({n_ok} ok, {n_bad} failed)")
    print(f"Wrote {program_out}")
    print(f"Wrote {fig_out}")
    if not summary.empty:
        head = summary.head(5)[["label", "wall_hours", "active_hours", "active_frac"]]
        print(head.to_string(index=False))


if __name__ == "__main__":
    main()
