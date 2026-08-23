"""Subprocess wrappers for MIDI-DDSP and DDSP-Piano (run in TF venv)."""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import tempfile
from pathlib import Path

import mido
import numpy as np
import soundfile as sf
import torch

from synthesis.ddsp.audio_convert import numpy_audio_to_stem_tensor
from synthesis.ddsp.config import (
    DDSP_PIANO_CKPT,
    DDSP_PIANO_GIN,
    DDSP_PIANO_ROOT,
    DDSP_PIANO_SAMPLE_RATE,
    DDSP_PIANO_TYPE,
    MIDI_DDSP_SAMPLE_RATE,
    MIDI_DDSP_WEIGHTS_DIR,
    PIPELINE_SAMPLE_RATE,
)
from synthesis.ddsp.env import DdspEnvError, ddsp_python_executable, ddsp_worker_env
from synthesis.ddsp.pool import ddsp_oneshot_enabled, get_ddsp_pool
from synthesis.ddsp.routing import BACKEND_DDSP_PIANO, BACKEND_MIDI_DDSP, StemRoute

# Cross-process lock for one-shot TF workers. Separate files so DDSP-Piano
# and MIDI-DDSP jobs can run in parallel. Override with SPDMX_DDSP_LOCK.
_DDSP_LOCK_DIR = Path(os.environ.get("SPDMX_DDSP_LOCK_DIR", "/tmp"))
_DDSP_LOCK_OVERRIDE = os.environ.get("SPDMX_DDSP_LOCK")

# Scale timeout by MIDI length (CPU is slow; GPU is faster but loads still take time).
_DDSP_TIMEOUT_BASE_SEC = float(os.environ.get("SPDMX_DDSP_TIMEOUT_BASE", "300"))
_DDSP_TIMEOUT_PER_AUDIO_SEC = float(os.environ.get("SPDMX_DDSP_TIMEOUT_PER_SEC", "20"))
_DDSP_TIMEOUT_MAX_SEC = float(os.environ.get("SPDMX_DDSP_TIMEOUT_MAX", "7200"))


def _midi_duration_sec(midi_path: Path) -> float:
    try:
        midi = mido.MidiFile(filename=str(midi_path), charset="utf8")
        return float(midi.length)
    except Exception:
        return 60.0


def _worker_timeout_sec(midi_path: Path) -> float:
    duration = _midi_duration_sec(midi_path)
    timeout = _DDSP_TIMEOUT_BASE_SEC + duration * _DDSP_TIMEOUT_PER_AUDIO_SEC
    return min(_DDSP_TIMEOUT_MAX_SEC, max(_DDSP_TIMEOUT_BASE_SEC, timeout))


def _ddsp_lock_path(backend: str | None) -> Path:
    if _DDSP_LOCK_OVERRIDE:
        return Path(_DDSP_LOCK_OVERRIDE)
    name = backend or "all"
    return _DDSP_LOCK_DIR / f"spdmx_ddsp_worker.{name}.lock"


def _run_worker(args: list[str], *, timeout_sec: float, backend: str | None = None) -> dict:
    """Legacy one-shot subprocess path (also used when SPDMX_DDSP_ONESHOT=1)."""
    python = ddsp_python_executable()
    cmd = [str(python), "-m", "synthesis.ddsp.worker", *args]
    env = ddsp_worker_env()
    lock_path = _ddsp_lock_path(backend)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                check=False,
                env=env,
            )
        except FileNotFoundError as exc:
            raise DdspEnvError(f"Failed to launch DDSP worker: {exc}") from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"DDSP worker timed out after {timeout_sec:.0f}s "
                f"(raise SPDMX_DDSP_TIMEOUT_PER_SEC or SPDMX_DDSP_TIMEOUT_MAX if needed)."
            ) from exc
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        hint = ""
        if proc.returncode in (-9, 137):
            hint = (
                " (process killed / SIGKILL — try SPDMX_DDSP_FORCE_CPU=1, or "
                "ensure nvidia CUDA-12 pip libs are installed in .venv-ddsp.)"
            )
        raise RuntimeError(
            f"DDSP worker failed (exit {proc.returncode}){hint}: {err[-2000:]}"
        )

    lines = [ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip()]
    if not lines:
        raise RuntimeError("DDSP worker produced no stdout status.")
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"DDSP worker returned non-JSON status: {lines[-1]!r}") from exc


def _log_ddsp_pool_failure(exc: BaseException, *, payload: dict) -> None:
    from synthesis.job_log import get_job_log

    log = get_job_log()
    if log is None:
        return
    msg = str(exc)
    stderr_tail = ""
    marker = "\nstderr:\n"
    if marker in msg:
        stderr_tail = msg.split(marker, 1)[1]
        msg = msg.split(marker, 1)[0]
    log.ddsp_failure(
        f"{msg} (command={payload.get('command')!r})",
        stderr_tail=stderr_tail,
    )


def _run_via_pool_or_oneshot(payload: dict, *, cli_args: list[str], timeout_sec: float) -> dict:
    if ddsp_oneshot_enabled():
        return _run_worker(
            cli_args,
            timeout_sec=timeout_sec,
            backend=str(payload.get("command") or ""),
        )
    pool = get_ddsp_pool()
    try:
        status = pool.submit(payload, timeout_sec=timeout_sec)
    except (RuntimeError, TimeoutError) as exc:
        _log_ddsp_pool_failure(exc, payload=payload)
        raise
    if not status.get("ok"):
        err = RuntimeError(
            f"DDSP pool job failed: {status.get('error', status)}"
        )
        _log_ddsp_pool_failure(err, payload=payload)
        raise err
    return status


def _midi_ddsp_weights_dir(weights_dir: Path | None) -> Path | None:
    weights = Path(weights_dir or MIDI_DDSP_WEIGHTS_DIR)
    if weights.is_dir() and any(weights.iterdir()):
        return weights
    return None


def synthesize_stem_midi_ddsp(
    midi_path: str | Path,
    instrument_name: str,
    *,
    weights_dir: Path | None = None,
) -> torch.Tensor:
    """Synthesize a monophonic stem with MIDI-DDSP; returns (channels, samples) @ 44.1 kHz."""
    midi_path = Path(midi_path)
    timeout = _worker_timeout_sec(midi_path)
    weights = _midi_ddsp_weights_dir(weights_dir)
    with tempfile.TemporaryDirectory(prefix="midi_ddsp_") as tmp:
        out_wav = Path(tmp) / "out.wav"
        cli_args = [
            "midi_ddsp",
            "--midi",
            str(midi_path),
            "--instrument",
            instrument_name,
            "--out",
            str(out_wav),
            *(["--weights-dir", str(weights)] if weights is not None else []),
        ]
        payload = {
            "command": "midi_ddsp",
            "midi": str(midi_path),
            "instrument": instrument_name,
            "out": str(out_wav),
            "weights_dir": str(weights) if weights is not None else "",
        }
        status = _run_via_pool_or_oneshot(
            payload, cli_args=cli_args, timeout_sec=timeout
        )
        sr = int(status.get("sample_rate", MIDI_DDSP_SAMPLE_RATE))
        audio, file_sr = sf.read(str(out_wav), always_2d=False)
        if file_sr:
            sr = int(file_sr)
        return numpy_audio_to_stem_tensor(
            np.asarray(audio, dtype=np.float32),
            source_sr=sr,
            target_sr=PIPELINE_SAMPLE_RATE,
        )


def synthesize_stem_ddsp_piano(
    midi_path: str | Path,
    *,
    piano_type: int | None = None,
    ckpt: Path | None = None,
    gin: Path | None = None,
    sample_rate: int | None = None,
) -> torch.Tensor:
    """Synthesize a (possibly polyphonic) piano stem with DDSP-Piano @ 44.1 kHz."""
    midi_path = Path(midi_path)
    timeout = _worker_timeout_sec(midi_path)
    piano_type_i = piano_type if piano_type is not None else DDSP_PIANO_TYPE
    ckpt_p = Path(ckpt or DDSP_PIANO_CKPT)
    gin_p = Path(gin or DDSP_PIANO_GIN)
    with tempfile.TemporaryDirectory(prefix="ddsp_piano_") as tmp:
        out_wav = Path(tmp) / "out.wav"
        cli_args = [
            "ddsp_piano",
            "--midi",
            str(midi_path),
            "--out",
            str(out_wav),
            "--piano-type",
            str(piano_type_i),
            "--ckpt",
            str(ckpt_p),
            "--gin",
            str(gin_p),
            "--piano-root",
            str(DDSP_PIANO_ROOT),
        ]
        payload = {
            "command": "ddsp_piano",
            "midi": str(midi_path),
            "out": str(out_wav),
            "piano_type": int(piano_type_i),
            "ckpt": str(ckpt_p),
            "gin": str(gin_p),
            "piano_root": str(DDSP_PIANO_ROOT),
        }
        status = _run_via_pool_or_oneshot(
            payload, cli_args=cli_args, timeout_sec=timeout
        )
        sr = int(status.get("sample_rate", sample_rate or DDSP_PIANO_SAMPLE_RATE))
        audio, file_sr = sf.read(str(out_wav), always_2d=False)
        if file_sr:
            sr = int(file_sr)
        return numpy_audio_to_stem_tensor(
            np.asarray(audio, dtype=np.float32),
            source_sr=sr,
            target_sr=PIPELINE_SAMPLE_RATE,
        )


def synthesize_stem_neural(midi_path: str | Path, route: StemRoute) -> torch.Tensor:
    """Dispatch to the neural backend indicated by ``route``."""
    if route.backend == BACKEND_MIDI_DDSP:
        if not route.instrument_key:
            raise ValueError("MIDI-DDSP route missing instrument_key")
        return synthesize_stem_midi_ddsp(midi_path, route.instrument_key)
    if route.backend == BACKEND_DDSP_PIANO:
        return synthesize_stem_ddsp_piano(midi_path)
    raise ValueError(f"Not a neural backend: {route.backend}")
