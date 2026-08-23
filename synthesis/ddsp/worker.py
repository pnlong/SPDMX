"""Worker entrypoint executed inside the isolated TF/DDSP venv.

Usage (from repo root, with SPDMX_DDSP_PYTHON or .venv-ddsp):

  .venv-ddsp/bin/python -m synthesis.ddsp.worker midi_ddsp \\
      --midi stem.mid --instrument violin --out out.wav --weights-dir ...

  .venv-ddsp/bin/python -m synthesis.ddsp.worker ddsp_piano \\
      --midi stem.mid --out out.wav --ckpt ... --gin ... --piano-root ...

  # Persistent JSONL server (one process per GPU; models cached):
  .venv-ddsp/bin/python -m synthesis.ddsp.worker serve
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def _write_wav(path: Path, audio, sample_rate: int) -> None:
    import numpy as np
    import soundfile as sf

    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim > 1:
        # Prefer mono file for pipeline simplicity.
        if arr.shape[0] <= 8 and arr.shape[0] < arr.shape[1]:
            arr = arr.mean(axis=0)
        else:
            arr = arr.mean(axis=-1) if arr.shape[-1] <= 8 else arr.reshape(-1)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), arr, sample_rate, subtype="FLOAT")


def _namespace(**kwargs: Any) -> SimpleNamespace:
    return SimpleNamespace(**kwargs)


# ---------------------------------------------------------------------------
# MIDI-DDSP (cached)
# ---------------------------------------------------------------------------

_MIDI_CACHE: dict[str, Any] = {
    "key": None,
    "synthesis_generator": None,
    "expression_generator": None,
}


def _midi_weights_key(weights_dir: str) -> str:
    return str(Path(weights_dir).resolve()) if weights_dir else ""


def _load_midi_ddsp_models(weights_dir: str):
    import time

    from synthesis.ddsp.midi_ddsp_ar_patch import apply_midi_ddsp_ar_patch

    apply_midi_ddsp_ar_patch()
    from midi_ddsp.midi_ddsp_synthesize import load_pretrained_model

    key = _midi_weights_key(weights_dir)
    if (
        _MIDI_CACHE["key"] == key
        and _MIDI_CACHE["synthesis_generator"] is not None
    ):
        return (
            _MIDI_CACHE["synthesis_generator"],
            _MIDI_CACHE["expression_generator"],
        )

    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    t0 = time.monotonic()
    sys.stderr.write(f"[ddsp gpu={gpu}] loading MIDI-DDSP weights...\n")
    sys.stderr.flush()

    load_kwargs: dict = {}
    weights = Path(weights_dir) if weights_dir else None
    if weights and weights.is_dir():
        synth_ckpt = weights / "synthesis_generator" / "50000"
        expr_ckpt = weights / "expression_generator" / "5000"
        nested = weights / "midi_ddsp_model_weights_urmp_9_10"
        if nested.is_dir():
            synth_ckpt = nested / "synthesis_generator" / "50000"
            expr_ckpt = nested / "expression_generator" / "5000"
        if synth_ckpt.exists():
            load_kwargs["synthesis_generator_path"] = str(synth_ckpt)
        if expr_ckpt.exists():
            load_kwargs["expression_generator_path"] = str(expr_ckpt)

    synthesis_generator, expression_generator = load_pretrained_model(**load_kwargs)
    _MIDI_CACHE["key"] = key
    _MIDI_CACHE["synthesis_generator"] = synthesis_generator
    _MIDI_CACHE["expression_generator"] = expression_generator
    sys.stderr.write(
        f"[ddsp gpu={gpu}] MIDI-DDSP weights ready in {time.monotonic() - t0:.0f}s\n"
    )
    sys.stderr.flush()
    return synthesis_generator, expression_generator


def _resolve_chunk_params(args: argparse.Namespace | SimpleNamespace) -> tuple[float, float]:
    """Shared chunk_sec / overlap_sec for MIDI-DDSP and DDSP-Piano."""
    chunk_sec = float(
        getattr(args, "chunk_sec", None)
        or os.environ.get("SPDMX_DDSP_CHUNK_SEC", "12")
    )
    overlap_raw = getattr(args, "overlap_sec", None)
    overlap_sec = float(
        overlap_raw
        if overlap_raw is not None
        else os.environ.get("SPDMX_DDSP_CHUNK_OVERLAP_SEC", "2.0")
    )
    if overlap_sec < 0:
        raise ValueError("overlap_sec must be >= 0")
    if overlap_sec >= chunk_sec:
        raise ValueError(
            f"overlap_sec ({overlap_sec}) must be smaller than chunk_sec ({chunk_sec})"
        )
    return chunk_sec, overlap_sec


# PrettyMIDI cannot round-trip notes shorter than ~1.5 ms; MIDI-DDSP uses 4 ms frames.
_MIN_CHUNK_NOTE_SEC = 0.004


def _slice_midi_time_window(
    midi_path: str | Path,
    start_sec: float,
    end_sec: float,
    out_path: Path,
) -> bool:
    """Write a PrettyMIDI slice for [start_sec, end_sec). Return False if empty."""
    import pretty_midi

    src = pretty_midi.PrettyMIDI(str(midi_path))
    dst = pretty_midi.PrettyMIDI()
    has_notes = False
    for inst in src.instruments:
        new_inst = pretty_midi.Instrument(
            program=inst.program,
            is_drum=inst.is_drum,
            name=inst.name,
        )
        for note in inst.notes:
            if note.end <= start_sec or note.start >= end_sec:
                continue
            clipped = pretty_midi.Note(
                velocity=note.velocity,
                pitch=note.pitch,
                start=max(0.0, note.start - start_sec),
                end=min(end_sec - start_sec, note.end - start_sec),
            )
            # Sub-frame slivers at chunk edges write invalid MIDI (0 instruments on read).
            if clipped.end > clipped.start + _MIN_CHUNK_NOTE_SEC:
                new_inst.notes.append(clipped)
                has_notes = True
        if new_inst.notes:
            dst.instruments.append(new_inst)
    if not has_notes:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dst.write(str(out_path))
    # Belt-and-suspenders: confirm PrettyMIDI can read what we wrote.
    check = pretty_midi.PrettyMIDI(str(out_path))
    if not check.instruments or not any(inst.notes for inst in check.instruments):
        return False
    return True


def _run_midi_ddsp(args: argparse.Namespace | SimpleNamespace) -> dict:
    """Run MIDI-DDSP in fixed-duration chunks with the same stitch logic as piano."""
    import tempfile

    import numpy as np
    import pretty_midi
    from midi_ddsp.data_handling.instrument_name_utils import INST_NAME_TO_ID_DICT
    from midi_ddsp.utils.midi_synthesis_utils import synthesize_mono_midi

    from synthesis.ddsp.chunking import plan_chunk_spans, stitch_audio_chunks

    synthesis_generator, expression_generator = _load_midi_ddsp_models(
        getattr(args, "weights_dir", "") or ""
    )
    instrument_id = INST_NAME_TO_ID_DICT[args.instrument]
    chunk_sec, overlap_sec = _resolve_chunk_params(args)
    sample_rate = 16000

    pm = pretty_midi.PrettyMIDI(str(args.midi))
    total_sec = float(pm.get_end_time())
    if total_sec <= 0:
        _write_wav(Path(args.out), np.zeros(0, dtype=np.float32), sample_rate)
        return {
            "ok": True,
            "backend": "midi_ddsp",
            "instrument": args.instrument,
            "sample_rate": sample_rate,
            "out": args.out,
        }

    total_samples = max(1, int(round(total_sec * sample_rate)))
    chunk_samples = max(1, int(round(chunk_sec * sample_rate)))
    overlap_samples = max(0, int(round(overlap_sec * sample_rate)))
    sample_spans = plan_chunk_spans(total_samples, chunk_samples, overlap_samples)

    audio_chunks: list[np.ndarray] = []
    with tempfile.TemporaryDirectory(prefix="midi_ddsp_chunk_") as tmp:
        tmp_dir = Path(tmp)
        for i, (start_sample, end_sample) in enumerate(sample_spans):
            keep = end_sample - start_sample
            start_sec = start_sample / sample_rate
            end_sec = end_sample / sample_rate
            chunk_midi = tmp_dir / f"chunk_{i}.mid"
            has_notes = _slice_midi_time_window(
                args.midi, start_sec, end_sec, chunk_midi
            )
            if not has_notes:
                audio_chunks.append(np.zeros(keep, dtype=np.float32))
                continue
            midi_audio, *_rest = synthesize_mono_midi(
                synthesis_generator,
                expression_generator,
                str(chunk_midi),
                instrument_id,
                output_dir=None,
                display_progressbar=False,
            )
            audio = np.asarray(midi_audio)
            if hasattr(audio, "numpy"):
                audio = audio.numpy()
            audio = np.squeeze(audio).astype(np.float32, copy=False)
            # Upstream may append ~1s of trailing rest; keep only this window.
            if audio.shape[0] >= keep:
                audio = audio[:keep]
            else:
                audio = np.pad(audio, (0, keep - audio.shape[0]))
            audio_chunks.append(audio)

    audio = stitch_audio_chunks(audio_chunks, sample_spans, overlap_samples)
    _write_wav(Path(args.out), audio, sample_rate)
    return {
        "ok": True,
        "backend": "midi_ddsp",
        "instrument": args.instrument,
        "sample_rate": sample_rate,
        "out": args.out,
    }


# ---------------------------------------------------------------------------
# DDSP-Piano (cached)
# ---------------------------------------------------------------------------

_PIANO_CACHE: dict[str, Any] = {
    "key": None,
    "model": None,
    "sample_rate": None,
    "chunk_sec": None,
    "warm_up": None,
}


def _piano_cache_key(
    *,
    piano_root: str,
    ckpt: str,
    gin: str,
    chunk_sec: float,
    warm_up: float,
) -> tuple:
    return (
        str(Path(piano_root).resolve()),
        str(Path(ckpt).resolve()),
        str(Path(gin).resolve()),
        float(chunk_sec),
        float(warm_up),
    )


def _load_ddsp_piano_model(
    *,
    piano_root: Path,
    ckpt: str,
    gin: str,
    chunk_sec: float,
    warm_up: float,
):
    import gin as gin_lib
    from ddsp.training import trainers, train_util
    from ddsp.training.models import get_model

    key = _piano_cache_key(
        piano_root=str(piano_root),
        ckpt=ckpt,
        gin=gin,
        chunk_sec=chunk_sec,
        warm_up=warm_up,
    )
    if _PIANO_CACHE["key"] == key and _PIANO_CACHE["model"] is not None:
        return _PIANO_CACHE["model"]

    if not piano_root.is_dir():
        raise FileNotFoundError(
            f"DDSP-Piano root not found: {piano_root}. "
            "Clone https://github.com/lrenault/ddsp-piano into "
            "synthesis/ddsp/third_party/ddsp-piano"
        )

    if str(piano_root) not in sys.path:
        sys.path.insert(0, str(piano_root))
    os.chdir(piano_root)

    from ddsp_piano.data_pipeline import get_dummy_data

    gin_lib.parse_config_file(str(gin))
    gin_lib.bind_parameter("%inference", True)
    gin_lib.bind_parameter("%duration", chunk_sec + warm_up)

    strategy = train_util.get_strategy()
    with strategy.scope():
        model = get_model()
        trainer = trainers.Trainer(model=model, strategy=strategy)
        trainer.build(
            get_dummy_data(
                batch_size=1,
                duration=chunk_sec + warm_up,
                sample_rate=model.sample_rate,
            )
        )
        trainer.restore(str(ckpt))

    _PIANO_CACHE["key"] = key
    _PIANO_CACHE["model"] = model
    _PIANO_CACHE["sample_rate"] = int(model.sample_rate)
    _PIANO_CACHE["chunk_sec"] = chunk_sec
    _PIANO_CACHE["warm_up"] = warm_up
    return model


def _run_ddsp_piano(args: argparse.Namespace | SimpleNamespace) -> dict:
    """Run DDSP-Piano inference in fixed-duration chunks (matches gin training)."""
    import numpy as np
    import tensorflow as tf
    from absl import logging
    from soundfile import write as sf_write

    piano_root = Path(args.piano_root)
    warm_up_raw = getattr(args, "warm_up", None)
    warm_up = float(warm_up_raw) if warm_up_raw is not None else 0.5
    chunk_sec, overlap_sec = _resolve_chunk_params(args)

    from synthesis.ddsp.chunking import (
        frames_to_samples,
        plan_chunk_spans,
        stitch_audio_chunks,
    )

    # Ensure piano package imports work before loading MIDI conditioning.
    if str(piano_root) not in sys.path:
        sys.path.insert(0, str(piano_root))
    os.chdir(piano_root)
    from ddsp_piano.utils.io_utils import load_midi_as_conditioning

    model = _load_ddsp_piano_model(
        piano_root=piano_root,
        ckpt=str(args.ckpt),
        gin=str(args.gin),
        chunk_sec=chunk_sec,
        warm_up=warm_up,
    )

    full = load_midi_as_conditioning(
        args.midi,
        duration=None,
        warm_up_duration=0.0,
    )
    cond = np.asarray(full["conditioning"])
    pedal = np.asarray(full["pedal"])
    frame_rate = 250
    chunk_frames = int(chunk_sec * frame_rate)
    overlap_frames = int(overlap_sec * frame_rate)
    warm_frames = int(warm_up * frame_rate)
    n_frames = cond.shape[1]
    frame_spans = plan_chunk_spans(n_frames, chunk_frames, overlap_frames)

    audio_chunks: list[np.ndarray] = []
    sample_spans: list[tuple[int, int]] = []
    for start, end in frame_spans:
        cond_chunk = cond[:, start:end]
        pedal_chunk = pedal[:, start:end]
        # Left-pad warm-up silence so recurrent state settles.
        if warm_frames > 0:
            cond_chunk = np.pad(
                cond_chunk,
                ((0, 0), (warm_frames, 0), (0, 0), (0, 0)),
                mode="constant",
            )
            pedal_chunk = np.pad(
                pedal_chunk,
                ((0, 0), (warm_frames, 0), (0, 0)),
                mode="constant",
            )
        # Right-pad to fixed model length.
        need = int((chunk_sec + warm_up) * frame_rate) - cond_chunk.shape[1]
        if need > 0:
            cond_chunk = np.pad(
                cond_chunk, ((0, 0), (0, need), (0, 0), (0, 0)), mode="constant"
            )
            pedal_chunk = np.pad(
                pedal_chunk, ((0, 0), (0, need), (0, 0)), mode="constant"
            )
        inputs = {
            "conditioning": tf.convert_to_tensor(cond_chunk, dtype=tf.float32),
            "pedal": tf.convert_to_tensor(pedal_chunk, dtype=tf.float32),
            "piano_model": tf.convert_to_tensor([[int(args.piano_type)]]),
        }
        outs = model(inputs)
        chunk_audio = outs["audio_synth"][0, int(warm_up * model.sample_rate) :].numpy()
        start_sample = frames_to_samples(start, frame_rate, model.sample_rate)
        end_sample = frames_to_samples(end, frame_rate, model.sample_rate)
        keep = end_sample - start_sample
        audio_chunks.append(chunk_audio[:keep])
        sample_spans.append((start_sample, end_sample))

    overlap_samples = frames_to_samples(overlap_frames, frame_rate, model.sample_rate)
    audio = stitch_audio_chunks(audio_chunks, sample_spans, overlap_samples)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf_write(str(out_path), data=np.asarray(audio, dtype=np.float32), samplerate=model.sample_rate)
    logging.info("DDSP-Piano wrote %s", out_path)
    return {
        "ok": True,
        "backend": "ddsp_piano",
        "piano_type": int(args.piano_type),
        "sample_rate": int(model.sample_rate),
        "out": str(out_path),
    }


# ---------------------------------------------------------------------------
# Persistent JSONL serve mode
# ---------------------------------------------------------------------------

def _request_to_args(req: dict) -> SimpleNamespace:
    command = req.get("command")
    if command == "midi_ddsp":
        return _namespace(
            midi=req["midi"],
            instrument=req["instrument"],
            out=req["out"],
            weights_dir=req.get("weights_dir", "") or "",
        )
    if command == "ddsp_piano":
        return _namespace(
            midi=req["midi"],
            out=req["out"],
            piano_type=int(req.get("piano_type", 0)),
            ckpt=req["ckpt"],
            gin=req["gin"],
            piano_root=req["piano_root"],
            warm_up=req.get("warm_up"),
            chunk_sec=req.get("chunk_sec"),
            overlap_sec=req.get("overlap_sec"),
        )
    raise ValueError(f"unknown command {command!r}")


def _handle_serve_request(req: dict) -> dict:
    req_id = req.get("id")
    command = req.get("command")
    if command == "ping":
        return {"id": req_id, "ok": True, "pong": True}
    if command == "shutdown":
        return {"id": req_id, "ok": True, "shutdown": True}
    try:
        args = _request_to_args(req)
        if command == "midi_ddsp":
            status = _run_midi_ddsp(args)
        elif command == "ddsp_piano":
            status = _run_ddsp_piano(args)
        else:
            raise ValueError(f"unknown command {command!r}")
        status = dict(status)
        status["id"] = req_id
        return status
    except Exception as exc:
        return {"id": req_id, "ok": False, "error": str(exc)}


def _emit(obj: dict) -> None:
    """Write one JSONL protocol line to stdout (never use print for logs here)."""
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def _run_serve() -> int:
    """Read JSONL jobs from stdin; write JSONL responses to stdout."""
    # Keep protocol on stdout; send incidental noise to stderr only.
    try:
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    except Exception:
        pass
    preload = os.environ.get("SPDMX_DDSP_PRELOAD", "").strip()
    if preload == "midi_ddsp":
        from synthesis.ddsp.config import MIDI_DDSP_WEIGHTS_DIR

        weights = str(MIDI_DDSP_WEIGHTS_DIR) if MIDI_DDSP_WEIGHTS_DIR.is_dir() else ""
        _load_midi_ddsp_models(weights)
    _emit({"ok": True, "ready": True})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as exc:
            _emit({"ok": False, "error": f"invalid JSON: {exc}"})
            continue
        status = _handle_serve_request(req)
        _emit(status)
        if req.get("command") == "shutdown":
            return 0
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="synthesis.ddsp.worker")
    sub = parser.add_subparsers(dest="command", required=True)

    p_midi = sub.add_parser("midi_ddsp")
    p_midi.add_argument("--midi", required=True)
    p_midi.add_argument("--instrument", required=True)
    p_midi.add_argument("--out", required=True)
    p_midi.add_argument("--weights-dir", default="")

    p_piano = sub.add_parser("ddsp_piano")
    p_piano.add_argument("--midi", required=True)
    p_piano.add_argument("--out", required=True)
    p_piano.add_argument("--piano-type", type=int, default=0)
    p_piano.add_argument("--ckpt", required=True)
    p_piano.add_argument("--gin", required=True)
    p_piano.add_argument("--piano-root", required=True)

    sub.add_parser("serve", help="Persistent JSONL server; models stay loaded")

    args = parser.parse_args(argv)
    if args.command == "serve":
        return _run_serve()

    try:
        if args.command == "midi_ddsp":
            status = _run_midi_ddsp(args)
        elif args.command == "ddsp_piano":
            status = _run_ddsp_piano(args)
        else:
            raise SystemExit(f"unknown command {args.command}")
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}), flush=True)
        print(f"worker error: {exc}", file=sys.stderr, flush=True)
        return 1

    print(json.dumps(status), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
