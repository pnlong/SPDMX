"""Evaluate trained Hybrid Demucs checkpoints with SI-SDR."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from experiments.separation.audio_io import load_mono, mono_to_stereo, sum_stems
from experiments.separation.medleydb_map import (
    bdgp_stem_files,
    iter_medleydb_track_dirs,
    load_track_metadata,
)
from experiments.separation.paths import (
    MEDLEYDB_METADATA_DIR,
    MEDLEYDB_ROOT,
    MOISESDB_ROOT,
    MUSDB_ROOT,
    TARGETS,
    TRAIN_ARMS,
    load_config,
    resolve_dev_dir,
)
from experiments.separation.sisdr import si_sdr
from experiments.separation.train import build_model

# Included in --write-paper CSV (figures may still subset).
PAPER_TEST_SETS = ("slakh2100", "spdmx_val", "musdb18", "medleydb", "moisesdb")
EVAL_SET_CHOICES = ("slakh", "spdmx_val", "musdb", "medleydb", "moisesdb")
MOISES_BDGP = ("bass", "drums", "guitar", "piano")


@torch.no_grad()
def separate_track(
    model: torch.nn.Module,
    mix_mono: np.ndarray,
    *,
    sample_rate: int,
    device: torch.device,
    segment: int = 441000,
) -> dict[str, np.ndarray]:
    """Run the model over a mono mix in overlapping chunks; return mono stems."""
    model.eval()
    hop = segment
    n = mix_mono.shape[0]
    sources = list(getattr(model, "sources", TARGETS))
    acc = {t: np.zeros(n, dtype=np.float64) for t in sources}
    weight = np.zeros(n, dtype=np.float64)
    for start in range(0, max(n, 1), hop):
        end = min(start + segment, n)
        chunk = mix_mono[start:end]
        if chunk.shape[0] < segment:
            chunk = np.pad(chunk, (0, segment - chunk.shape[0]))
        mix_t = torch.from_numpy(mono_to_stereo(chunk)).unsqueeze(0).to(device)
        est = model(mix_t)[0].detach().cpu().numpy()  # (S, 2, T)
        for i, t in enumerate(sources):
            mono = est[i].mean(axis=0)[: end - start]
            acc[t][start:end] += mono
        weight[start:end] += 1.0
    weight = np.maximum(weight, 1e-8)
    return {t: (acc[t] / weight).astype(np.float32) for t in sources}


def eval_pack_manifest(
    model: torch.nn.Module,
    manifest_csv: Path,
    packs_root: Path,
    *,
    test_set: str,
    sample_rate: int,
    device: torch.device,
    desc: str | None = None,
) -> pd.DataFrame:
    """SI-SDR on packed 4-stem songs listed in a manifest CSV."""
    rows = []
    df = pd.read_csv(manifest_csv)
    for _, row in tqdm(df.iterrows(), total=len(df), desc=desc or f"eval:{test_set}"):
        song_dir = packs_root / row["path"]
        refs = {}
        for t in TARGETS:
            refs[t], _ = load_mono(song_dir / f"{t}.flac", sample_rate=sample_rate)
        mix, _ = load_mono(song_dir / "mix.flac", sample_rate=sample_rate)
        est = separate_track(model, mix, sample_rate=sample_rate, device=device)
        for t in TARGETS:
            rows.append(
                {
                    "test_set": test_set,
                    "song_id": row["song_id"],
                    "target": t,
                    "si_sdr": si_sdr(est[t], refs[t]),
                    "mixture_si_sdr": si_sdr(mix, refs[t]),
                }
            )
    return pd.DataFrame(rows)


def eval_slakh_test(
    model: torch.nn.Module,
    test_csv: Path,
    packs_root: Path,
    *,
    sample_rate: int,
    device: torch.device,
) -> pd.DataFrame:
    return eval_pack_manifest(
        model,
        test_csv,
        packs_root,
        test_set="slakh2100",
        sample_rate=sample_rate,
        device=device,
        desc="eval:slakh",
    )


def eval_musdb_bass_drums(
    model: torch.nn.Module,
    musdb_root: Path,
    *,
    sample_rate: int,
    device: torch.device,
) -> pd.DataFrame:
    """Cross-domain: only Bass/Drums (shared with MUSDB)."""
    test_dir = musdb_root / "test"
    if not test_dir.is_dir():
        # musdb18hq layout sometimes flat under test/
        alt = musdb_root / "musdb18hq" / "test"
        test_dir = alt if alt.is_dir() else test_dir
    if not test_dir.is_dir():
        print(f"MUSDB test not found at {musdb_root}; skipping")
        return pd.DataFrame()

    rows = []
    tracks = sorted(p for p in test_dir.iterdir() if p.is_dir())
    for track in tqdm(tracks, desc="eval:musdb"):
        bass_p = track / "bass.wav"
        drums_p = track / "drums.wav"
        mix_p = track / "mixture.wav"
        if not (bass_p.is_file() and drums_p.is_file() and mix_p.is_file()):
            continue
        bass, _ = load_mono(bass_p, sample_rate=sample_rate)
        drums, _ = load_mono(drums_p, sample_rate=sample_rate)
        mix, _ = load_mono(mix_p, sample_rate=sample_rate)
        est = separate_track(model, mix, sample_rate=sample_rate, device=device)
        for t, ref in (("bass", bass), ("drums", drums)):
            rows.append(
                {
                    "test_set": "musdb18",
                    "song_id": track.name,
                    "target": t,
                    "si_sdr": si_sdr(est[t], ref),
                    "mixture_si_sdr": si_sdr(mix, ref),
                }
            )
    return pd.DataFrame(rows)


def eval_medleydb_bdgp(
    model: torch.nn.Module,
    medleydb_root: Path,
    *,
    metadata_dir: Path,
    sample_rate: int,
    device: torch.device,
) -> pd.DataFrame:
    """Cross-domain SI-SDR on MedleyDB V1+V2 for present BDGP targets."""
    if not medleydb_root.is_dir():
        print(f"MedleyDB not found at {medleydb_root}; skipping")
        return pd.DataFrame()
    if not metadata_dir.is_dir():
        print(f"MedleyDB metadata not found at {metadata_dir}; skipping")
        return pd.DataFrame()

    rows = []
    tracks = list(iter_medleydb_track_dirs(medleydb_root))
    for track_id, track_dir in tqdm(tracks, desc="eval:medleydb"):
        meta = load_track_metadata(metadata_dir, track_id)
        if not meta:
            continue
        if str(meta.get("has_bleed", "no")).strip().lower() in {"yes", "true", "1"}:
            # Bleed makes stem refs impure; skip for SI-SDR transfer checks.
            continue
        mix_name = meta.get("mix_filename") or f"{track_id}_MIX.wav"
        mix_path = track_dir / mix_name
        if not mix_path.is_file():
            continue
        stem_dir_name = meta.get("stem_dir") or f"{track_id}_STEMS"
        stem_dir = track_dir / stem_dir_name
        if not stem_dir.is_dir():
            continue
        target_files = bdgp_stem_files(meta)
        if not target_files:
            continue
        refs: dict[str, np.ndarray] = {}
        for target, filenames in target_files.items():
            parts = []
            for fn in filenames:
                p = stem_dir / fn
                if not p.is_file():
                    parts = []
                    break
                audio, _ = load_mono(p, sample_rate=sample_rate)
                parts.append(audio)
            if parts:
                refs[target] = sum_stems(parts) if len(parts) > 1 else parts[0]
        if not refs:
            continue
        ref_len = max(r.shape[0] for r in refs.values())
        mix: np.ndarray | None = None
        try:
            cand, _ = load_mono(mix_path, sample_rate=sample_rate)
            # Some MedleyDB MIX files are mislabeled AAC/ALAC and only partially decode.
            if cand.shape[0] >= max(1, int(0.5 * ref_len)):
                mix = cand
            else:
                print(
                    f"MedleyDB {track_id}: MIX decode too short "
                    f"({cand.shape[0]} < 0.5*{ref_len}); using stem sum"
                )
        except Exception as exc:  # noqa: BLE001
            print(f"MedleyDB {track_id}: MIX unreadable ({exc}); using stem sum")
        if mix is None:
            all_parts = []
            for p in sorted(stem_dir.glob(f"{track_id}_STEM_*.wav")):
                audio, _ = load_mono(p, sample_rate=sample_rate)
                all_parts.append(audio)
            if not all_parts:
                print(f"MedleyDB {track_id}: no stems to rebuild MIX; skipping")
                continue
            mix = sum_stems(all_parts)
        est = separate_track(model, mix, sample_rate=sample_rate, device=device)
        for t, ref in refs.items():
            n = min(est[t].shape[0], ref.shape[0], mix.shape[0])
            rows.append(
                {
                    "test_set": "medleydb",
                    "song_id": track_id,
                    "target": t,
                    "si_sdr": si_sdr(est[t][:n], ref[:n]),
                    "mixture_si_sdr": si_sdr(mix[:n], ref[:n]),
                }
            )
    return pd.DataFrame(rows)


def _moises_to_mono(audio: np.ndarray) -> np.ndarray:
    """Normalize MoisesDB arrays to mono float32 (T,)."""
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim == 1:
        return x
    if x.ndim == 2:
        # (C, T) or (T, C)
        if x.shape[0] <= 8 and x.shape[0] < x.shape[1]:
            return x.mean(axis=0)
        return x.mean(axis=1)
    raise ValueError(f"unexpected MoisesDB audio shape {x.shape}")


def _resolve_moises_data_path(moises_root: Path) -> Path | None:
    """Accept MoisesDB/ or MoisesDB/moisesdb_v0.1 as data_path."""
    if not moises_root.is_dir():
        return None
    nested = moises_root / "moisesdb_v0.1"
    if nested.is_dir() and any(nested.iterdir()):
        return moises_root
    # Already pointing at moisesdb_v0.1
    if moises_root.name == "moisesdb_v0.1" and any(moises_root.iterdir()):
        return moises_root.parent
    # Empty placeholder root → soft-skip
    if not any(moises_root.iterdir()):
        return None
    return moises_root


def eval_moisesdb_bdgp(
    model: torch.nn.Module,
    moises_root: Path,
    *,
    sample_rate: int,
    device: torch.device,
) -> pd.DataFrame:
    """Cross-domain SI-SDR on MoisesDB for present BDGP top-level stems."""
    data_path = _resolve_moises_data_path(moises_root)
    if data_path is None:
        print(f"MoisesDB not found or empty at {moises_root}; skipping")
        return pd.DataFrame()
    try:
        from moisesdb.dataset import MoisesDB
    except ImportError:
        print("moisesdb package not installed; skipping MoisesDB eval")
        return pd.DataFrame()

    try:
        db = MoisesDB(data_path=str(data_path), sample_rate=sample_rate)
    except Exception as exc:  # noqa: BLE001 — soft-skip until layout is confirmed
        print(f"MoisesDB failed to open at {data_path}: {exc}; skipping")
        return pd.DataFrame()

    rows = []
    n = len(db)
    for i in tqdm(range(n), desc="eval:moisesdb"):
        track = db[i]
        try:
            stems = track.stems or {}
            mix = _moises_to_mono(track.audio)
        except Exception as exc:  # noqa: BLE001
            print(f"MoisesDB track {getattr(track, 'id', i)} load failed: {exc}")
            continue
        present = {
            t: _moises_to_mono(stems[t])
            for t in MOISES_BDGP
            if t in stems and stems[t] is not None
        }
        if not present:
            continue
        est = separate_track(model, mix, sample_rate=sample_rate, device=device)
        song_id = str(getattr(track, "id", None) or getattr(track, "name", i))
        for t, ref in present.items():
            n_samp = min(est[t].shape[0], ref.shape[0], mix.shape[0])
            rows.append(
                {
                    "test_set": "moisesdb",
                    "song_id": song_id,
                    "target": t,
                    "si_sdr": si_sdr(est[t][:n_samp], ref[:n_samp]),
                    "mixture_si_sdr": si_sdr(mix[:n_samp], ref[:n_samp]),
                }
            )
    return pd.DataFrame(rows)


def load_checkpoint(ckpt_path: Path, device: torch.device):
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    sources = list(blob.get("sources") or TARGETS)
    sample_rate = int(blob.get("sample_rate") or 44100)
    model = build_model(sources, sample_rate)
    model.load_state_dict(blob["model"])
    model.to(device)
    model.sources = sources  # type: ignore[attr-defined]
    return model, sample_rate, blob.get("arm", "unknown")


def _atomic_to_csv(df: pd.DataFrame, path: Path) -> None:
    """Write CSV atomically so parallel merges never read a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            df.to_csv(f, index=False)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _summarize(per_track: pd.DataFrame) -> pd.DataFrame:
    return (
        per_track.groupby(["train_arm", "test_set", "target"], as_index=False)["si_sdr"]
        .mean()
        .rename(columns={"si_sdr": "si_sdr_mean"})
    )


def _merge_arm_csvs(out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Concatenate per-arm CSVs written by (possibly parallel) eval runs."""
    paths = sorted(out_dir.glob("si_sdr_per_track_*.csv"))
    if not paths:
        raise FileNotFoundError(f"no per-arm eval CSVs under {out_dir}")
    frames = [pd.read_csv(p) for p in paths]
    full = pd.concat(frames, ignore_index=True)
    # Last write wins if an arm was re-run (drop duplicate song/target rows).
    full = full.drop_duplicates(
        subset=["train_arm", "test_set", "song_id", "target"], keep="last"
    )
    return full, _summarize(full)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        choices=(*TRAIN_ARMS, "all"),
        default="all",
    )
    parser.add_argument(
        "--eval-sets",
        nargs="+",
        choices=EVAL_SET_CHOICES,
        default=["slakh", "spdmx_val", "musdb", "medleydb", "moisesdb"],
        help=(
            "Which test sets to score. slakh=Slakh2100 test; "
            "spdmx_val=sPDMX val packs; musdb=MUSDB18 bass/drums; "
            "medleydb=MedleyDB V1+V2 BDGP; moisesdb=MoisesDB BDGP (soft-skip if missing). "
            "Default: all five."
        ),
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--musdb-root", type=Path, default=None)
    parser.add_argument("--medleydb-root", type=Path, default=None)
    parser.add_argument("--medleydb-metadata", type=Path, default=None)
    parser.add_argument("--moisesdb-root", type=Path, default=None)
    parser.add_argument(
        "--write-paper",
        action="store_true",
        help=(
            "Also write separation_sisdr.csv under the eval dir on SPDMX_OUTPUT_DIR "
            "(includes slakh2100, spdmx_val, musdb18, medleydb, moisesdb)."
        ),
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Skip inference; only merge existing per-arm CSVs into combined outputs.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    root = resolve_dev_dir(cfg)
    packs_root = root / "packs"
    manifests = root / "manifests"
    out_dir = root / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    musdb_root = args.musdb_root or Path(cfg.get("musdb_root") or MUSDB_ROOT)
    medleydb_root = args.medleydb_root or Path(cfg.get("medleydb_root") or MEDLEYDB_ROOT)
    metadata_dir = args.medleydb_metadata or Path(
        cfg.get("medleydb_metadata") or MEDLEYDB_METADATA_DIR
    )
    moises_root = args.moisesdb_root or Path(cfg.get("moisesdb_root") or MOISESDB_ROOT)
    eval_sets = set(args.eval_sets)

    if not args.merge_only:
        arms = TRAIN_ARMS if args.arm == "all" else (args.arm,)
        print(f"eval: train arms={list(arms)}  eval_sets={sorted(eval_sets)}")
        for arm in arms:
            ckpt_dir = root / "checkpoints" / arm
            # Prefer best val, then best train, then last.
            for name in ("best_val.ckpt", "best_train.ckpt", "last.ckpt", "best.pt", "final.pt"):
                cand = ckpt_dir / name
                if cand.is_file():
                    ckpt = cand
                    break
            else:
                print(f"missing checkpoint for {arm} under {ckpt_dir}")
                continue
            print(f"{arm}: evaluating {ckpt.name} on {sorted(eval_sets)}")
            model, sr, _ = load_checkpoint(ckpt, device)
            arm_rows: list[pd.DataFrame] = []

            if "slakh" in eval_sets:
                # Shared Slakh2100 test (same file for every train arm).
                test_csv = manifests / "slakh" / "test.csv"
                if not test_csv.is_file():
                    test_csv = manifests / arm / "test.csv"
                if test_csv.is_file():
                    df = eval_pack_manifest(
                        model,
                        test_csv,
                        packs_root,
                        test_set="slakh2100",
                        sample_rate=sr,
                        device=device,
                        desc="eval:slakh",
                    )
                    df["train_arm"] = arm
                    arm_rows.append(df)
                else:
                    print(f"{arm}: missing Slakh test CSV; skipping slakh eval set")

            if "spdmx_val" in eval_sets:
                val_csv = manifests / "spdmx" / "val.csv"
                if val_csv.is_file():
                    df = eval_pack_manifest(
                        model,
                        val_csv,
                        packs_root,
                        test_set="spdmx_val",
                        sample_rate=sr,
                        device=device,
                        desc="eval:spdmx_val",
                    )
                    df["train_arm"] = arm
                    arm_rows.append(df)
                else:
                    print(f"missing {val_csv}; skipping spdmx_val")

            if "musdb" in eval_sets:
                mus = eval_musdb_bass_drums(model, musdb_root, sample_rate=sr, device=device)
                if len(mus):
                    mus["train_arm"] = arm
                    arm_rows.append(mus)

            if "medleydb" in eval_sets:
                med = eval_medleydb_bdgp(
                    model,
                    medleydb_root,
                    metadata_dir=metadata_dir,
                    sample_rate=sr,
                    device=device,
                )
                if len(med):
                    med["train_arm"] = arm
                    arm_rows.append(med)

            if "moisesdb" in eval_sets:
                moi = eval_moisesdb_bdgp(
                    model, moises_root, sample_rate=sr, device=device
                )
                if len(moi):
                    moi["train_arm"] = arm
                    arm_rows.append(moi)

            if not arm_rows:
                print(f"{arm}: no eval rows; check --eval-sets and manifests")
                continue

            arm_new = pd.concat(arm_rows, ignore_index=True)
            arm_path = out_dir / f"si_sdr_per_track_{arm}.csv"
            # Preserve other test sets when re-running a subset (e.g. medleydb only).
            if arm_path.is_file():
                prev = pd.read_csv(arm_path)
                new_sets = set(arm_new["test_set"].astype(str).unique())
                keep = prev[~prev["test_set"].astype(str).isin(new_sets)]
                arm_full = pd.concat([keep, arm_new], ignore_index=True)
            else:
                arm_full = arm_new
            arm_full = arm_full.drop_duplicates(
                subset=["train_arm", "test_set", "song_id", "target"], keep="last"
            )
            arm_summary = _summarize(arm_full)
            # Per-arm files: parallel --arm runs do not overwrite each other.
            _atomic_to_csv(arm_full, arm_path)
            _atomic_to_csv(arm_summary, out_dir / f"si_sdr_summary_{arm}.csv")
            print(arm_summary.to_string(index=False))

    try:
        full, summary = _merge_arm_csvs(out_dir)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc

    full_path = out_dir / "si_sdr_per_track.csv"
    summary_path = out_dir / "si_sdr_summary.csv"
    _atomic_to_csv(full, full_path)
    _atomic_to_csv(summary, summary_path)

    paper_csv = out_dir / "separation_sisdr.csv"
    paper_path: str | None = None
    if args.write_paper:
        paper = summary[summary["test_set"].astype(str).isin(PAPER_TEST_SETS)].copy()
        _atomic_to_csv(paper, paper_csv)
        paper_path = str(paper_csv)

    print(
        json.dumps(
            {
                "per_track": str(full_path),
                "summary": str(summary_path),
                "per_arm": sorted(str(p) for p in out_dir.glob("si_sdr_per_track_*.csv")),
                "paper": paper_path,
            },
            indent=2,
        )
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
