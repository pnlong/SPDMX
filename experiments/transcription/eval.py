"""Evaluate transcription arms with per-track metrics + bootstrap CIs.

YourMT3 ``test.py`` is file-wise; after our patches it appends one JSONL row
per track. This wrapper:

1. Symlinks deepfreeze checkpoints into ``amt/logs/transcription/<arm>/``
2. Runs ``test.py`` with multi-GPU + larger inference sub-batches
3. Merges per-rank JSONLs and computes percentile bootstrap CIs
4. Optionally writes ``analysis/paper_data/transcription_note_f1.csv``

Overnight (4 GPUs; SPDMX→SPDMX gets two GPUs, Slakh-test jobs one each):

    uv run python -m experiments.transcription.eval --all --gpu 0,1,2,3 --parallel --write-paper

Single job:

    uv run python -m experiments.transcription.eval \\
      --train-arm spdmx --test-preset spdmx --gpu 0,1 --write-paper

Recompute CIs only (no GPU):

    uv run python -m experiments.transcription.eval --merge-only --write-paper
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

from experiments.transcription.metrics_ci import merge_rank_jsonls, summarize_per_track
from experiments.transcription.paths import REPO_ROOT, TRANS_DIR, load_config, resolve_dev_dir

YOURMT3_SRC = TRANS_DIR / "YourMT3" / "amt" / "src"
YOURMT3_LOGS = TRANS_DIR / "YourMT3" / "amt" / "logs"
REPO_PAPER = REPO_ROOT / "analysis" / "paper_data"

# (train_arm, test_preset, paper test_set label)
PAPER_JOBS = (
    ("slakh", "slakh_redux", "slakh"),
    ("spdmx", "slakh_redux", "slakh"),
    ("spdmx", "spdmx", "spdmx"),
)


def _ensure_ckpt_link(arm: str, ckpt_root: Path) -> Path:
    """Point amt/logs/transcription/<arm> at deepfreeze checkpoints/<arm>."""
    src = ckpt_root / arm
    ckpt_dir = src / "checkpoints"
    if not ckpt_dir.is_dir():
        raise SystemExit(f"missing checkpoints for arm={arm}: {ckpt_dir}")
    last = ckpt_dir / "last.ckpt"
    if not last.is_file():
        step = ckpt_dir / "epoch=26-step=100000.ckpt"
        if not step.is_file():
            raise SystemExit(f"no last.ckpt under {ckpt_dir}")
        if last.is_symlink() or last.exists():
            last.unlink()
        last.symlink_to(step.name)
        print(f"symlinked {last} → {step.name}")
    dest_parent = YOURMT3_LOGS / "transcription"
    dest_parent.mkdir(parents=True, exist_ok=True)
    dest = dest_parent / arm
    if dest.is_symlink() or dest.exists():
        if dest.is_symlink() and dest.resolve() == src.resolve():
            return dest
        if dest.is_symlink() or dest.is_file():
            dest.unlink()
        else:
            raise SystemExit(f"refusing to replace non-symlink {dest}")
    dest.symlink_to(src)
    print(f"symlinked {dest} → {src}")
    return dest


def _job_dir(dev: Path, train_arm: str, test_preset: str) -> Path:
    return dev / "eval" / f"{train_arm}__{test_preset}"


def _run_yourmt3_test(
    *,
    train_arm: str,
    test_preset: str,
    gpu: str,
    num_gpus: str,
    precision: str,
    strategy: str,
    per_track_dir: Path,
    subbsz: int,
    pack_target_segs: int,
    num_workers: int,
    dry_run: bool,
) -> int:
    from experiments.transcription.patch_yourmt3 import apply_yourmt3_patches

    for p in apply_yourmt3_patches(YOURMT3_SRC):
        print(f"patched {p}")

    n_vis = len([x for x in gpu.split(",") if x.strip() != ""]) if gpu else 1
    if num_gpus == "auto":
        num_gpus = str(max(1, n_vis))

    use_ddp = int(num_gpus) > 1
    cmd = [
        sys.executable,
        str(YOURMT3_SRC / "test.py"),
        train_arm,
        "-p",
        "transcription",
        "-d",
        test_preset,
        "-tk",
        "mt3_full_plus",
        "-pr",
        precision,
        "-st",
        "ddp" if use_ddp else strategy,
        "-g",
        num_gpus,
        "-wb",
        "disabled",
        "-pt",
        "False",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{YOURMT3_SRC.resolve()}:{env.get('PYTHONPATH', '')}"
    if gpu:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    env["SPDMX_PER_TRACK_DIR"] = str(per_track_dir.resolve())
    env["SPDMX_TEST_SUBBSZ"] = str(subbsz)
    if pack_target_segs > 0:
        env["SPDMX_PACK_TARGET_SEGS"] = str(pack_target_segs)
    else:
        env.pop("SPDMX_PACK_TARGET_SEGS", None)
    # test_step never uses GT tokens; skipping tokenize keeps the GPU fed
    env.setdefault("SPDMX_EVAL_SKIP_TOKENS", "1")
    env["SPDMX_TEST_NUM_WORKERS"] = str(num_workers)
    env["SPDMX_TEST_PREFETCH"] = env.get("SPDMX_TEST_PREFETCH", "4")
    env["SPDMX_TEST_PERSISTENT_WORKERS"] = "1"
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")

    print("cwd:", YOURMT3_SRC)
    print("cmd:", " ".join(cmd))
    print("CUDA_VISIBLE_DEVICES=", env.get("CUDA_VISIBLE_DEVICES"))
    print("SPDMX_PER_TRACK_DIR=", env["SPDMX_PER_TRACK_DIR"])
    print("SPDMX_TEST_SUBBSZ=", env["SPDMX_TEST_SUBBSZ"])
    print("SPDMX_PACK_TARGET_SEGS=", env.get("SPDMX_PACK_TARGET_SEGS", "(off)"))
    if dry_run:
        return 0

    per_track_dir.mkdir(parents=True, exist_ok=True)
    for old in per_track_dir.glob("per_track_rank*.jsonl"):
        old.unlink()
    return subprocess.call(cmd, cwd=str(YOURMT3_SRC), env=env)


def _summarize_job(job_dir: Path, *, n_boot: int, seed: int) -> dict:
    from experiments.transcription.metrics_ci import load_per_track_rows

    merged = merge_rank_jsonls(job_dir)
    df = load_per_track_rows(merged)
    summary = summarize_per_track(df, n_boot=n_boot, seed=seed)
    out = {
        "n_tracks": int(len(df)),
        "metrics": summary,
        "per_track_jsonl": str(merged),
    }
    (job_dir / "summary.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"summary {job_dir}: n={out['n_tracks']} onset={summary.get('onset_f')} multi={summary.get('multi_f')}")
    return out


def _paper_rows_from_summaries(dev: Path) -> pd.DataFrame:
    rows = []
    for train_arm, test_preset, test_set in PAPER_JOBS:
        summary_path = _job_dir(dev, train_arm, test_preset) / "summary.json"
        if not summary_path.is_file():
            rows.append(
                {
                    "arm": train_arm,
                    "test_set": test_set,
                    "note_f1_onset": None,
                    "note_f1_onset_ci_half": None,
                    "multi_f1": None,
                    "multi_f1_ci_half": None,
                    "n_tracks": None,
                    "status": "pending",
                }
            )
            continue
        s = json.loads(summary_path.read_text())
        onset = s["metrics"].get("onset_f", {})
        multi = s["metrics"].get("multi_f", {})
        rows.append(
            {
                "arm": train_arm,
                "test_set": test_set,
                "note_f1_onset": onset.get("mean"),
                "note_f1_onset_ci_lo": onset.get("ci_lo"),
                "note_f1_onset_ci_hi": onset.get("ci_hi"),
                "note_f1_onset_ci_half": onset.get("ci_half"),
                "multi_f1": multi.get("mean"),
                "multi_f1_ci_lo": multi.get("ci_lo"),
                "multi_f1_ci_hi": multi.get("ci_hi"),
                "multi_f1_ci_half": multi.get("ci_half"),
                "n_tracks": s.get("n_tracks"),
                "status": "ok",
            }
        )
    return pd.DataFrame(rows)


def _write_paper_csv(dev: Path) -> Path:
    df = _paper_rows_from_summaries(dev)
    paper = REPO_PAPER / "transcription_note_f1.csv"
    paper.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(paper, index=False)
    out = dev / "metrics" / "transcription_note_f1.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {paper}")
    print(f"wrote {out}")
    print(df.to_string(index=False))
    return paper


def _gpu_waves(gpu: str) -> tuple[list[list[tuple[str, str, str]]], dict[tuple[str, str], str]]:
    """Waves of concurrent jobs + (train, preset) → CUDA_VISIBLE_DEVICES."""
    ids = [g.strip() for g in gpu.split(",") if g.strip() != ""]
    if not ids:
        raise SystemExit("--parallel requires --gpu")
    big = ("spdmx", "spdmx", "spdmx")
    a = ("slakh", "slakh_redux", "slakh")
    b = ("spdmx", "slakh_redux", "slakh")
    assign: dict[tuple[str, str], str] = {}
    if len(ids) >= 4:
        assign[("spdmx", "spdmx")] = f"{ids[0]},{ids[1]}"
        assign[("slakh", "slakh_redux")] = ids[2]
        assign[("spdmx", "slakh_redux")] = ids[3]
        return [[big, a, b]], assign
    if len(ids) == 3:
        assign[("spdmx", "spdmx")] = f"{ids[0]},{ids[1]}"
        assign[("slakh", "slakh_redux")] = ids[2]
        assign[("spdmx", "slakh_redux")] = ids[2]
        return [[big, a], [b]], assign
    if len(ids) == 2:
        assign[("spdmx", "spdmx")] = f"{ids[0]},{ids[1]}"
        assign[("slakh", "slakh_redux")] = ids[0]
        assign[("spdmx", "slakh_redux")] = ids[1]
        return [[big], [a, b]], assign
    for j in (big, a, b):
        assign[(j[0], j[1])] = ids[0]
    return [[big], [a], [b]], assign


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-arm", choices=("slakh", "spdmx"), default=None)
    parser.add_argument("--test-preset", choices=("slakh_redux", "spdmx"), default=None)
    parser.add_argument("--all", action="store_true", help="Run the three paper jobs")
    parser.add_argument("--gpu", type=str, default=None, help="CUDA_VISIBLE_DEVICES, e.g. 0,1,2,3")
    parser.add_argument("--num-gpus", type=str, default="auto", help="Lightning -g (default: # visible)")
    parser.add_argument("--precision", type=str, default="bf16-mixed")
    parser.add_argument("--strategy", type=str, default="auto")
    parser.add_argument("--subbsz", type=int, default=256, help="Per-GPU inference chunk (SPDMX_TEST_SUBBSZ)")
    parser.add_argument(
        "--pack-target-segs",
        type=int,
        default=256,
        help="Pack songs until ~N segments/step (SPDMX_PACK_TARGET_SEGS); 0 disables",
    )
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader workers per GPU")
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--ckpt-root", type=Path, default=None, help="Default: {dev}/checkpoints")
    parser.add_argument("--parallel", action="store_true", help="Run paper jobs in GPU-disjoint waves")
    parser.add_argument("--merge-only", action="store_true", help="Only merge JSONL + write CIs / paper CSV")
    parser.add_argument("--write-paper", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)

    if not (YOURMT3_SRC / "test.py").is_file():
        raise SystemExit(
            f"YourMT3 missing at {YOURMT3_SRC}. Run: uv run python -m experiments.transcription.setup_yourmt3"
        )

    cfg = load_config(args.config)
    dev = resolve_dev_dir(cfg)
    ckpt_root = args.ckpt_root or (dev / "checkpoints")

    if args.merge_only:
        for train_arm, test_preset, _ in PAPER_JOBS:
            job = _job_dir(dev, train_arm, test_preset)
            if any(job.glob("per_track*.jsonl")):
                _summarize_job(job, n_boot=args.n_boot, seed=args.seed)
        if args.write_paper:
            _write_paper_csv(dev)
        return

    if args.all:
        jobs = list(PAPER_JOBS)
    else:
        if args.train_arm is None or args.test_preset is None:
            parser.error("provide --train-arm and --test-preset, or --all / --merge-only")
        lab = "slakh" if args.test_preset == "slakh_redux" else "spdmx"
        jobs = [(args.train_arm, args.test_preset, lab)]

    if args.parallel:
        if not args.all:
            parser.error("--parallel requires --all")
        if not args.gpu:
            parser.error("--parallel requires --gpu")
        waves, assign = _gpu_waves(args.gpu)
        rc = 0
        for wave in waves:
            procs = []
            for train_arm, test_preset, _ in wave:
                g = assign[(train_arm, test_preset)]
                cmd = [
                    sys.executable,
                    "-m",
                    "experiments.transcription.eval",
                    "--train-arm",
                    train_arm,
                    "--test-preset",
                    test_preset,
                    "--gpu",
                    g,
                    "--num-gpus",
                    "auto",
                    "--precision",
                    args.precision,
                    "--subbsz",
                    str(args.subbsz),
                    "--pack-target-segs",
                    str(args.pack_target_segs),
                    "--num-workers",
                    str(args.num_workers),
                    "--n-boot",
                    str(args.n_boot),
                    "--seed",
                    str(args.seed),
                    "--ckpt-root",
                    str(ckpt_root),
                ]
                print("spawn:", " ".join(cmd))
                if args.dry_run:
                    continue
                _ensure_ckpt_link(train_arm, ckpt_root)
                job_dir = _job_dir(dev, train_arm, test_preset)
                job_dir.mkdir(parents=True, exist_ok=True)
                log_path = job_dir / "eval.log"
                print("  log →", log_path)
                log_f = open(log_path, "w")
                procs.append(
                    (
                        (train_arm, test_preset),
                        subprocess.Popen(
                            cmd,
                            cwd=str(REPO_ROOT),
                            stdout=log_f,
                            stderr=subprocess.STDOUT,
                            env=os.environ.copy(),
                        ),
                        log_f,
                    )
                )
            if args.dry_run:
                continue
            for key, proc, log_f in procs:
                code = proc.wait()
                log_f.close()
                print(f"finished {key} rc={code}")
                rc = rc or code
                if code == 0:
                    _summarize_job(_job_dir(dev, key[0], key[1]), n_boot=args.n_boot, seed=args.seed)
        if args.write_paper:
            _write_paper_csv(dev)
        raise SystemExit(rc)

    rc = 0
    for train_arm, test_preset, _ in jobs:
        _ensure_ckpt_link(train_arm, ckpt_root)
        job_dir = _job_dir(dev, train_arm, test_preset)
        code = _run_yourmt3_test(
            train_arm=train_arm,
            test_preset=test_preset,
            gpu=args.gpu or "",
            num_gpus=args.num_gpus,
            precision=args.precision,
            strategy=args.strategy,
            per_track_dir=job_dir,
            subbsz=args.subbsz,
            pack_target_segs=args.pack_target_segs,
            num_workers=args.num_workers,
            dry_run=args.dry_run,
        )
        rc = rc or code
        if code == 0 and not args.dry_run:
            lightning = YOURMT3_LOGS / "transcription" / train_arm
            for src in lightning.glob(f"result_*_{test_preset}.json"):
                shutil.copy2(src, job_dir / src.name)
            _summarize_job(job_dir, n_boot=args.n_boot, seed=args.seed)
    if args.write_paper:
        _write_paper_csv(dev)
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
