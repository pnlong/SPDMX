"""Realify raw fluidsynth stems using Stable Audio 3 audio-to-audio."""

from __future__ import annotations

import argparse
import logging
import multiprocessing
import os
import queue
import shutil
import sys
import threading
import warnings
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from shared.config import (
    ABLATION_SAMPLE_SEED,
    DEFAULT_AUDIO_FORMAT,
    OUTPUT_DIR,
    REALIFY_BATCH_SIZE,
    REALIFY_BACKEND,
    REALIFY_CFG_SCALE,
    REALIFY_CHUNKED_DECODE,
    REALIFY_CONTENT_FIDELITY_ENFORCE,
    REALIFY_CONTENT_FIDELITY_MAX_ATTEMPTS,
    REALIFY_CONTENT_FIDELITY_MIN_NOISE,
    REALIFY_CONTENT_FIDELITY_NOISE_STEP,
    REALIFY_INIT_NOISE_LEVEL,
    REALIFY_MIN_GPU_FREE_GB,
    REALIFY_SILENCE_ENFORCE,
    REALIFY_STEPS,
    SAMPLE_RATE,
)
from synthesis.audio import (
    ensure_stem_channels,
    load_stem,
    stem_duration_seconds,
    stem_is_valid,
    stem_n_samples,
    stem_path,
    write_audio,
)
from synthesis.paths import full_stems_dir, remap_path_prefix, resolve_output_song_dir
from synthesis.realify.captions.generate import generate_captions
from synthesis.realify.chunking import (
    max_realify_chunk_samples,
    needs_chunking,
    plan_chunk_spans,
    realify_overlap_samples,
    stitch_chunk_outputs,
)
from synthesis.realify.preset_config import (
    DEFAULT_PRESETS_FILE,
    load_presets,
    preset_key,
    resolve_category,
    select_preset,
)
from synthesis.realify.silence import apply_silence_enforcement

logger = logging.getLogger(__name__)

_REALIFY_MODEL = None
_REALIFY_PRESETS: dict | None = None
_REALIFY_BATCH_SIZE = REALIFY_BATCH_SIZE
_REALIFY_WORKER_CONFIG: dict | None = None
_REALIFY_PROGRESS_QUEUE = None
_REALIFY_PROGRESS_SENTINEL = "__realify_progress_done__"


def _report_realify_progress(n: int) -> None:
    if _REALIFY_PROGRESS_QUEUE is not None and n:
        _REALIFY_PROGRESS_QUEUE.put(n)


def _run_pool_with_stem_progress(
    pool,
    shard_args: list,
    *,
    total_tasks: int,
    desc: str,
    worker_fn,
) -> None:
    """Run shard workers and update tqdm as each realify batch finishes."""
    progress_queue = _REALIFY_PROGRESS_QUEUE
    if progress_queue is None:
        raise RuntimeError("Progress queue was not initialized for worker pool")

    def drain_progress(progress_bar: tqdm) -> None:
        while True:
            try:
                item = progress_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if item == _REALIFY_PROGRESS_SENTINEL:
                break
            progress_bar.update(int(item))

    with tqdm(total=total_tasks, desc=desc, unit="stem") as progress:
        drain_thread = threading.Thread(
            target=drain_progress,
            args=(progress,),
            daemon=True,
        )
        drain_thread.start()
        try:
            for _ in pool.imap(worker_fn, shard_args, chunksize=1):
                pass
        finally:
            progress_queue.put(_REALIFY_PROGRESS_SENTINEL)
            drain_thread.join(timeout=30)


def stem_seed(sample_seed: int, song_path: str, track: int) -> int:
    return (sample_seed + hash((song_path, track))) % (2**31)


def build_generate_kwargs(
    *,
    preset: dict,
    model,
    prompt: str | list[str],
    duration_seconds: float | list[float],
    init_audio,
    seed: int | list[int],
    batch_size: int = 1,
) -> dict:
    kwargs = {
        "init_audio": init_audio,
        "init_noise_level": preset.get("init_noise_level", REALIFY_INIT_NOISE_LEVEL),
        "prompt": prompt,
        "duration": duration_seconds,
        "steps": preset.get("steps", REALIFY_STEPS),
        "cfg_scale": preset.get("cfg_scale", REALIFY_CFG_SCALE),
        "seed": seed,
        "batch_size": batch_size,
        "sample_size": model.model_config["sample_size"],
        "disable_tqdm": True,
        "chunked_decode": REALIFY_CHUNKED_DECODE,
    }
    negative_prompt = preset.get("negative_prompt")
    if negative_prompt is not None:
        kwargs["negative_prompt"] = negative_prompt
    return kwargs


def task_preset(task: dict, presets: dict) -> dict:
    if "preset" in task:
        return dict(task["preset"])
    row = pd.Series(task["row"])
    return select_preset(presets, row)


def task_needs_chunking(task: dict, model) -> bool:
    return needs_chunking(int(task["duration"] * SAMPLE_RATE), model)


def task_n_samples(task: dict) -> int | None:
    if "n_samples" in task:
        return int(task["n_samples"])
    stem_path = task.get("stem_path")
    if stem_path:
        return stem_n_samples(Path(stem_path))
    return None


def task_category(task: dict, presets: dict) -> str:
    row = pd.Series(task["row"])
    return resolve_category(row, presets)


def sort_realify_tasks_for_batching(tasks: list[dict], presets: dict) -> list[dict]:
    """Order stems category-first, then by length (minimizes pad within batches)."""

    def sort_key(task: dict):
        row = task.get("row") or {}
        path = str(row.get("path", task.get("stem_path", "")))
        track = int(row.get("track", 0) or 0)
        return (
            task_category(task, presets),
            task_n_samples(task) or 0,
            path,
            track,
        )

    return sorted(tasks, key=sort_key)


def shard_tasks_contiguous(tasks: list[dict], n_workers: int) -> list[list[dict]]:
    """Split into contiguous blocks (keeps category/length locality for batching)."""
    if n_workers <= 0:
        raise ValueError("n_workers must be >= 1")
    if n_workers == 1:
        return [list(tasks)]
    n = len(tasks)
    if n == 0:
        return [[] for _ in range(n_workers)]
    base, rem = divmod(n, n_workers)
    shards: list[list[dict]] = []
    start = 0
    for i in range(n_workers):
        size = base + (1 if i < rem else 0)
        shards.append(list(tasks[start : start + size]))
        start += size
    return shards


def _pad_waveforms_to_common_length(
    waveforms: list[torch.Tensor],
) -> tuple[list[torch.Tensor], list[int]]:
    lengths = [int(w.shape[-1]) for w in waveforms]
    max_len = max(lengths)
    if len(set(lengths)) == 1:
        return waveforms, lengths
    padded = [
        torch.nn.functional.pad(w, (0, max_len - w.shape[-1]))
        for w in waveforms
    ]
    return padded, lengths


def _trim_waveform(waveform: torch.Tensor, n_samples: int) -> torch.Tensor:
    return waveform[..., :n_samples]


def iter_realify_batches(
    tasks: list[dict],
    model,
    presets: dict,
    batch_size: int,
):
    """Yield task groups that can share one SA3 forward pass.

    Groups by preset only (not length). Callers should category-sort then
    length-sort so neighbors share a preset and pad waste stays small;
    ``realify_stems_batch`` pads to the batch max length and trims after.
    Chunked stems stay singleton batches.
    """
    if batch_size <= 1:
        for task in tasks:
            yield [task]
        return

    buffer: list[dict] = []
    buffer_key: tuple | None = None

    def flush():
        nonlocal buffer, buffer_key
        if buffer:
            yield_now = buffer
            buffer = []
            buffer_key = None
            return yield_now
        return None

    for task in tasks:
        preset = task_preset(task, presets)
        key = preset_key(preset)
        if task_needs_chunking(task, model):
            pending = flush()
            if pending is not None:
                yield pending
            yield [task]
            continue

        if buffer and (len(buffer) >= batch_size or key != buffer_key):
            yield buffer
            buffer = []
            buffer_key = None

        buffer.append(task)
        buffer_key = key

    if buffer:
        yield buffer


def visible_cuda_count() -> int:
    try:
        import torch
        return torch.cuda.device_count()
    except ImportError:
        return 0


def gpu_memory_snapshot(device_index: int) -> tuple[float, float, str]:
    """Return (free_gb, total_gb, device_name) for a visible CUDA device."""
    import torch

    free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
    return (
        free_bytes / (1024**3),
        total_bytes / (1024**3),
        torch.cuda.get_device_name(device_index),
    )


def suggest_realify_batch_size(
    free_gb: float,
    total_gb: float,
    *,
    model: str = "medium",
    max_batch: int = 4,
) -> int:
    """Heuristic SA3 batch size from free/total VRAM.

    Caps by GPU class (total) and by current free memory so a busy 3090
    does not over-subscribe. Tuned for medium; small-music is lighter.
    """
    if model == "small-music":
        base_gb = 4.0
        per_extra_gb = 1.5
        # (total_gb upper bound exclusive-ish, max batch)
        total_caps = ((8.0, 1), (12.0, 2), (16.0, 3), (float("inf"), 4))
    else:
        # medium: ~8–10 GiB for batch=1; ~3 GiB per extra concurrent stem.
        base_gb = 9.0
        per_extra_gb = 3.0
        total_caps = ((13.0, 1), (20.0, 2), (26.0, 3), (float("inf"), 4))

    by_free = 1
    if free_gb >= base_gb and per_extra_gb > 0:
        by_free = 1 + int((free_gb - base_gb) / per_extra_gb)

    by_total = 1
    for threshold, cap in total_caps:
        if total_gb < threshold:
            by_total = cap
            break

    return max(1, min(int(max_batch), int(by_free), int(by_total)))


def resolve_realify_batch_size(
    requested: int | None,
    *,
    free_gb: float,
    total_gb: float,
    model: str = "medium",
) -> int:
    """Return explicit batch size, or VRAM heuristic when requested is None/<=0."""
    if requested is not None and int(requested) > 0:
        return int(requested)
    return suggest_realify_batch_size(free_gb, total_gb, model=model)


def select_realify_gpu_indices(
    *,
    min_free_gb: float = REALIFY_MIN_GPU_FREE_GB,
    log_skips: bool = True,
) -> list[int]:
    """Pick visible GPUs with enough free VRAM for an SA3 worker."""
    count = visible_cuda_count()
    if count == 0:
        return []

    selected: list[int] = []
    for device_index in range(count):
        free_gb, total_gb, name = gpu_memory_snapshot(device_index)
        if free_gb >= min_free_gb:
            selected.append(device_index)
            continue
        if log_skips:
            print(
                f"Realify: skipping GPU {device_index} ({name}) — "
                f"{free_gb:.1f} GiB free of {total_gb:.1f} GiB "
                f"(need {min_free_gb:.1f} GiB free per worker)"
            )
    return selected


def describe_visible_gpus(*, min_free_gb: float | None = None) -> str:
    count = visible_cuda_count()
    if count == 0:
        return "no CUDA devices visible"
    try:
        import torch

        parts = []
        for device_index in range(count):
            free_gb, total_gb, name = gpu_memory_snapshot(device_index)
            parts.append(f"{name} ({free_gb:.1f}/{total_gb:.1f} GiB free)")
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "all")
        summary = f"{count} GPU(s): {', '.join(parts)} (CUDA_VISIBLE_DEVICES={visible})"
        if min_free_gb is not None:
            usable = select_realify_gpu_indices(min_free_gb=min_free_gb, log_skips=False)
            summary += f"; {len(usable)} usable at >= {min_free_gb:.1f} GiB free"
        return summary
    except Exception:
        return f"{count} GPU(s)"


def reset_realify_output(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def log_realify_plan(
    *,
    source_dir: Path,
    output_dir: Path,
    model: str,
    n_tasks: int,
    n_captions: int,
    use_gpu: bool,
    batch_size: int = REALIFY_BATCH_SIZE,
) -> None:
    backend = "GPU" if use_gpu else "CPU"
    print(f"Realify source: {source_dir}")
    print(f"Realify output: {output_dir}")
    print(f"Realify model: {model} ({backend})")
    if batch_size is None or int(batch_size) <= 0:
        print("Realify batch size: auto (per-GPU from free/total VRAM)")
    elif batch_size > 1:
        print(f"Realify batch size: {batch_size} stems per forward pass")
    else:
        print("Realify batch size: 1 stem per forward pass")
    if use_gpu:
        print(
            f"Realify devices: {describe_visible_gpus(min_free_gb=REALIFY_MIN_GPU_FREE_GB)}"
        )
    else:
        print("Realify devices: CPU workers (no visible CUDA GPU)")
    print(f"Realify stems queued: {n_tasks} of {n_captions}")
    if n_tasks == 0:
        print(
            "Realify: all stems already exist in the output tree; skipping SA3."
        )


def realify_uses_gpu(model: str) -> bool:
    """Return True when realify should run on visible CUDA devices."""
    cuda_count = visible_cuda_count()
    if model == "medium":
        if cuda_count == 0:
            raise RuntimeError(
                "SA3 medium requires a GPU. Set CUDA_VISIBLE_DEVICES to select device(s), "
                "or use -m small-music for CPU realify."
            )
        return True
    # small-music: prefer GPU when visible, otherwise CPU multiprocessing
    return cuda_count > 0


def should_use_flash_attention(device_index: int | None = None) -> bool:
    """FlashAttention requires Ampere (sm_80) or newer."""
    try:
        import torch
    except ImportError:
        return False
    if not torch.cuda.is_available():
        return False
    if device_index is None:
        device_index = torch.cuda.current_device()
    major, _ = torch.cuda.get_device_capability(device_index)
    return major >= 8


def _patch_sa3_disable_flash_attention() -> None:
    import stable_audio_3.models.transformer as transformer_mod

    transformer_mod.flash_attn_func = None
    transformer_mod.flash_attn_kvpacked_func = None
    transformer_mod.flash_attn_varlen_func = None
    # Avoid torch.compile flex_attention attempts on the SDPA fallback path.
    transformer_mod.flex_attention_compiled = None


def sa3_repo_path() -> Path:
    """Path to the stable-audio-3 git submodule."""
    return Path(__file__).resolve().parent / "stable-audio-3"


def configure_sa3_env() -> None:
    """Set process env and import path before loading SA3 (including spawn workers)."""
    sa3_path = sa3_repo_path()
    if sa3_path.is_dir():
        path_str = str(sa3_path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    os.environ.setdefault("TORCH_LOGS", "-dynamo,-inductor,-dynamic")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


def _silence_sa3_loggers() -> None:
    for logger_name in (
        "torch",
        "torch._dynamo",
        "torch._inductor",
        "torch.fx",
        "transformers",
        "huggingface_hub",
    ):
        logging.getLogger(logger_name).setLevel(logging.ERROR)


def configure_sa3_runtime(*, device_index: int | None = None) -> None:
    """Silence noisy SA3 load warnings and disable FlashAttention when unsupported."""
    configure_sa3_env()
    _silence_sa3_loggers()
    warnings.filterwarnings(
        "ignore",
        message=r".*weight_norm.*",
        category=FutureWarning,
    )
    warnings.filterwarnings("ignore", module=r"torch\._dynamo.*")
    if should_use_flash_attention(device_index):
        return

    _patch_sa3_disable_flash_attention()
    try:
        import torch
    except ImportError:
        print("Realify: CUDA unavailable; using SA3 SDPA attention fallback.")
        return

    if device_index is None:
        if torch.cuda.is_available():
            device_index = torch.cuda.current_device()
        else:
            print("Realify: CUDA unavailable; using SA3 SDPA attention fallback.")
            return

    major, minor = torch.cuda.get_device_capability(device_index)
    print(
        f"Realify: GPU {device_index} (sm {major}.{minor}) does not support "
        "FlashAttention; using SA3 SDPA attention fallback."
    )


def load_model(model_name: str, device_index: int | None = None):
    configure_sa3_runtime(device_index=device_index)
    from stable_audio_3 import StableAudioModel
    import torch

    if device_index is not None:
        device = f"cuda:{device_index}"
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    return StableAudioModel.from_pretrained(model_name, device=device)


def _normalize_generated_audio(audio) -> torch.Tensor:
    """Return stem-format float32 tensor (channels, samples) on CPU."""
    if hasattr(audio, "ndim") and audio.ndim == 3:
        audio = audio[0]
    return ensure_stem_channels(audio)


def load_realify_model(
    model_name: str,
    *,
    backend: str = REALIFY_BACKEND,
    device_index: int | None = None,
):
    """Load PyTorch SA3 or a persistent TensorRT session."""
    if backend == "trt":
        from synthesis.realify.trt_backend import TrtRealifyError, get_trt_session

        try:
            return get_trt_session(model_name=model_name, steps=REALIFY_STEPS)
        except TrtRealifyError as exc:
            raise RuntimeError(str(exc)) from exc
    return load_model(model_name, device_index=device_index)


def _is_trt_model(model) -> bool:
    return type(model).__name__ == "TrtRealifySession"


def _generate_realify_audio(
    *,
    preset: dict,
    model,
    prompt: str | list[str],
    duration_seconds: float | list[float],
    init_audio,
    seed: int | list[int],
    batch_size: int = 1,
) -> torch.Tensor:
    if _is_trt_model(model):
        if batch_size != 1:
            raise ValueError("TensorRT backend does not support batch_size > 1")
        if isinstance(prompt, list) or isinstance(duration_seconds, list):
            raise ValueError("TensorRT backend does not support batched prompts/durations")
        sr, waveform = init_audio
        del sr
        return model.generate(
            waveform=waveform,
            prompt=str(prompt),
            duration_seconds=float(duration_seconds),
            init_noise_level=float(preset.get("init_noise_level", REALIFY_INIT_NOISE_LEVEL)),
            seed=int(seed),
            cfg_scale=float(preset.get("cfg_scale", REALIFY_CFG_SCALE)),
        )

    kwargs = build_generate_kwargs(
        preset=preset,
        model=model,
        prompt=prompt,
        duration_seconds=duration_seconds,
        init_audio=init_audio,
        seed=seed,
        batch_size=batch_size,
    )
    try:
        audio = model.generate(**kwargs)
    except torch.cuda.OutOfMemoryError:
        import torch

        torch.cuda.empty_cache()
        kwargs["chunked_decode"] = True
        audio = model.generate(**kwargs)
    return audio


def _normalize_generated_batch(audio) -> list[torch.Tensor]:
    if hasattr(audio, "ndim") and audio.ndim == 3:
        return [_normalize_generated_audio(audio[i : i + 1]) for i in range(audio.shape[0])]
    return [_normalize_generated_audio(audio)]


def _generate_and_enforce(
    *,
    reference: torch.Tensor,
    preset: dict,
    model,
    prompt: str,
    init_audio,
    duration_seconds: float,
    seed: int,
    silence_enforce: bool,
) -> torch.Tensor:
    audio = _generate_realify_audio(
        preset=preset,
        model=model,
        prompt=prompt,
        init_audio=init_audio,
        duration_seconds=duration_seconds,
        seed=seed,
    )
    return apply_silence_enforcement(
        reference,
        _normalize_generated_audio(audio),
        enabled=silence_enforce,
    )


def _realify_with_content_fidelity_backoff(
    *,
    reference: torch.Tensor,
    preset: dict,
    model,
    prompt: str,
    init_audio,
    duration_seconds: float,
    seed: int,
    silence_enforce: bool,
    output_path: Path,
) -> torch.Tensor:
    from synthesis.realify.content_fidelity import score_content_fidelity

    noise = float(preset.get("init_noise_level", REALIFY_INIT_NOISE_LEVEL))
    step = REALIFY_CONTENT_FIDELITY_NOISE_STEP
    min_noise = REALIFY_CONTENT_FIDELITY_MIN_NOISE
    last_result = None

    for attempt in range(1, REALIFY_CONTENT_FIDELITY_MAX_ATTEMPTS + 1):
        trial_preset = {**preset, "init_noise_level": noise}
        audio = _generate_and_enforce(
            reference=reference,
            preset=trial_preset,
            model=model,
            prompt=prompt,
            init_audio=init_audio,
            duration_seconds=duration_seconds,
            seed=seed,
            silence_enforce=silence_enforce,
        )
        last_result = score_content_fidelity(reference, audio)
        logger.info(
            "Content fidelity %s attempt %d/%d noise=%.2f score=%.3f "
            "extra=%d missing=%d passed=%s",
            output_path.name,
            attempt,
            REALIFY_CONTENT_FIDELITY_MAX_ATTEMPTS,
            noise,
            last_result.score,
            last_result.extra_onsets,
            last_result.missing_onsets,
            last_result.passed,
        )
        if last_result.passed:
            return audio

        noise -= step
        if noise < min_noise - 1e-9:
            break

    logger.info(
        "Content fidelity fallback to reference for %s (last score=%.3f)",
        output_path,
        last_result.score if last_result is not None else float("nan"),
    )
    return reference


def realify_stem(
    init_audio_path: Path,
    output_path: Path,
    prompt: str,
    preset: dict,
    model,
    duration_seconds: float,
    seed: int,
    audio_format: str = DEFAULT_AUDIO_FORMAT,
    silence_enforce: bool = REALIFY_SILENCE_ENFORCE,
    content_fidelity_enforce: bool = REALIFY_CONTENT_FIDELITY_ENFORCE,
):
    waveform = load_stem(init_audio_path)
    total_samples = waveform.shape[-1]
    chunk_samples = max_realify_chunk_samples(model)
    overlap_samples = realify_overlap_samples()
    allow_fidelity_backoff = content_fidelity_enforce and not needs_chunking(total_samples, model)
    if content_fidelity_enforce and needs_chunking(total_samples, model):
        logger.warning(
            "Content fidelity backoff disabled for chunked stem %s",
            init_audio_path,
        )

    if not needs_chunking(total_samples, model):
        init_audio = (SAMPLE_RATE, waveform)
        if allow_fidelity_backoff:
            audio = _realify_with_content_fidelity_backoff(
                reference=waveform,
                preset=preset,
                model=model,
                prompt=prompt,
                init_audio=init_audio,
                duration_seconds=duration_seconds,
                seed=seed,
                silence_enforce=silence_enforce,
                output_path=output_path,
            )
        else:
            audio = _generate_and_enforce(
                reference=waveform,
                preset=preset,
                model=model,
                prompt=prompt,
                init_audio=init_audio,
                duration_seconds=duration_seconds,
                seed=seed,
                silence_enforce=silence_enforce,
            )
        write_audio(audio, output_path, audio_format)
        return

    spans = plan_chunk_spans(total_samples, chunk_samples, overlap_samples)
    chunks = []
    for chunk_index, (start, end) in enumerate(spans):
        chunk_waveform = waveform[..., start:end]
        chunk_duration = (end - start) / SAMPLE_RATE
        chunk_seed = (seed + chunk_index) % (2**31)
        chunks.append(
            _generate_and_enforce(
                reference=chunk_waveform,
                preset=preset,
                model=model,
                prompt=prompt,
                init_audio=(SAMPLE_RATE, chunk_waveform),
                duration_seconds=chunk_duration,
                seed=chunk_seed,
                silence_enforce=silence_enforce,
            )
        )

    stitched = stitch_chunk_outputs(chunks, spans, overlap_samples)
    if content_fidelity_enforce:
        from synthesis.realify.content_fidelity import score_content_fidelity

        result = score_content_fidelity(waveform, stitched)
        logger.info(
            "Content fidelity (chunked, no retry) %s score=%.3f passed=%s",
            output_path.name,
            result.score,
            result.passed,
        )
    write_audio(stitched, output_path, audio_format)
    return


def realify_stems_batch(
    tasks: list[dict],
    *,
    model,
    presets: dict,
    audio_format: str,
    silence_enforce: bool = REALIFY_SILENCE_ENFORCE,
    content_fidelity_enforce: bool = REALIFY_CONTENT_FIDELITY_ENFORCE,
) -> None:
    """Realify multiple stems in one SA3 forward pass."""
    if len(tasks) == 1:
        task = tasks[0]
        row = pd.Series(task["row"])
        realify_stem(
            init_audio_path=Path(task["stem_path"]),
            output_path=Path(task["out_path"]),
            prompt=row["prompt"],
            preset=task_preset(task, presets),
            model=model,
            duration_seconds=task["duration"],
            seed=task["seed"],
            audio_format=audio_format,
            silence_enforce=silence_enforce,
            content_fidelity_enforce=content_fidelity_enforce,
        )
        return

    if content_fidelity_enforce:
        raise RuntimeError("Content fidelity enforce requires batch size 1")

    rows = [pd.Series(task["row"]) for task in tasks]
    preset = task_preset(tasks[0], presets)
    waveforms = [load_stem(Path(task["stem_path"])) for task in tasks]
    waveforms, original_lengths = _pad_waveforms_to_common_length(waveforms)
    # Match SA3 duration to the padded waveform length so shorter stems
    # get enough samples before trim.
    max_duration = max(original_lengths) / float(SAMPLE_RATE)
    audio = _generate_realify_audio(
        preset=preset,
        model=model,
        prompt=[row["prompt"] for row in rows],
        duration_seconds=[max_duration] * len(tasks),
        init_audio=[(SAMPLE_RATE, waveform) for waveform in waveforms],
        seed=[task["seed"] for task in tasks],
        batch_size=len(tasks),
    )
    for task, stem_audio, n_samples, reference in zip(
        tasks,
        _normalize_generated_batch(audio),
        original_lengths,
        waveforms,
    ):
        stem_audio = _trim_waveform(stem_audio, n_samples)
        stem_audio = apply_silence_enforcement(
            _trim_waveform(reference, n_samples),
            stem_audio,
            enabled=silence_enforce,
        )
        write_audio(
            stem_audio,
            Path(task["out_path"]),
            audio_format,
        )


def process_realify_tasks(
    tasks: list[dict],
    *,
    model,
    presets: dict,
    audio_format: str,
    batch_size: int,
    desc: str,
    show_progress: bool = True,
    silence_enforce: bool = REALIFY_SILENCE_ENFORCE,
    content_fidelity_enforce: bool = REALIFY_CONTENT_FIDELITY_ENFORCE,
    backend: str = REALIFY_BACKEND,
) -> None:
    if _is_trt_model(model) and batch_size != 1:
        batch_size = 1
    if content_fidelity_enforce and batch_size != 1:
        batch_size = 1
    tasks = sort_realify_tasks_for_batching(tasks, presets)
    batches = list(iter_realify_batches(tasks, model, presets, batch_size))
    progress = tqdm(total=len(tasks), desc=desc, unit="stem", disable=not show_progress)
    try:
        for batch in batches:
            realify_stems_batch(
                batch,
                model=model,
                presets=presets,
                audio_format=audio_format,
                silence_enforce=silence_enforce,
                content_fidelity_enforce=content_fidelity_enforce,
            )
            progress.update(len(batch))
            _report_realify_progress(len(batch))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        progress.close()


def resolve_stem_output_path(
    song_dir: Path,
    track: int,
    source_dir: Path,
    output_dir: Path,
    audio_format: str = DEFAULT_AUDIO_FORMAT,
) -> Path:
    out_song_dir = resolve_output_song_dir(song_dir, source_dir, output_dir)
    return stem_path(out_song_dir, track, audio_format)


def copy_metadata_tables(source_dir: Path, output_dir: Path):
    """Copy data/stems/routing CSVs into ``output_dir``, remapping absolute song paths."""
    from synthesis.ddsp.config import DDSP_ROUTING_FILE_NAME
    from shared.config import DATA_DIR_NAME

    if source_dir.resolve() == output_dir.resolve():
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in (f"{DATA_DIR_NAME}.csv", "stems.csv", DDSP_ROUTING_FILE_NAME, "stem_recipe.csv"):
        src = source_dir / name
        if not src.exists():
            continue
        table = pd.read_csv(src)
        if "path" in table.columns:
            table["path"] = table["path"].map(
                lambda p: remap_path_prefix(str(p), source_dir, output_dir)
            )
        table.to_csv(output_dir / name, index=False)


def _stem_metadata_row(
    captions_row: pd.Series,
    stems_df: pd.DataFrame | None,
) -> pd.Series:
    if stems_df is None or stems_df.empty:
        return captions_row
    match = stems_df[
        (stems_df["path"] == captions_row["path"])
        & (stems_df["track"] == captions_row["track"])
    ]
    if match.empty:
        return captions_row
    merged = match.iloc[0].to_dict()
    merged.update(captions_row.to_dict())
    return pd.Series(merged)


def _load_ddsp_routing_index(source_dir: Path) -> dict[tuple[str, int], dict]:
    """Map (path, track) → routing row for DDSP donor reuse."""
    from synthesis.ddsp.config import DDSP_ROUTING_FILE_NAME

    routing_path = source_dir / DDSP_ROUTING_FILE_NAME
    if not routing_path.is_file():
        return {}
    df = pd.read_csv(routing_path)
    index: dict[tuple[str, int], dict] = {}
    for _, row in df.iterrows():
        index[(str(row["path"]), int(row["track"]))] = row.to_dict()
    return index


def build_realify_tasks(
    captions: pd.DataFrame,
    source_dir: Path,
    output_dir: Path,
    audio_format: str = DEFAULT_AUDIO_FORMAT,
    sample_seed: int = ABLATION_SAMPLE_SEED,
    presets: dict | None = None,
    *,
    output_root: str | None = None,
    render_mode: str | None = None,
    category_allowlist: set[str] | frozenset[str] | None = None,
    recipe=None,
    reset: bool = False,
) -> list[dict]:
    from synthesis.realify.preset_config import realify_enabled, select_preset
    from synthesis.reuse import (
        copy_stem,
        donor_mode_from_source,
        donor_realify_stem_path,
        fallback_donor_mode,
        is_reused_source,
        song_rel_under_data,
        uses_ddsp,
    )

    stems_path = source_dir / "stems.csv"
    stems_df = pd.read_csv(stems_path) if stems_path.is_file() else None
    routing_index = _load_ddsp_routing_index(source_dir)
    donor_mode = fallback_donor_mode(render_mode) if render_mode else None
    ddsp_mode = bool(render_mode and uses_ddsp(render_mode))
    dest_recipe_index = {}
    if recipe is not None:
        from synthesis.recipe import load_stem_recipe_index

        dest_recipe_index = load_stem_recipe_index(output_dir)

    tasks = []
    original_path_updates: list[tuple[str, int, str]] = []
    for _, row in captions.iterrows():
        song_dir = Path(row["path"])
        track = int(row["track"])
        out_path = resolve_stem_output_path(
            song_dir, track, source_dir, output_dir, audio_format,
        )
        if out_path.exists() and not reset:
            if recipe is None:
                continue
            from synthesis.recipe import (
                desired_realify_fingerprint,
                listening_category_from_stem_row,
                recorded_realify_fingerprint,
            )

            out_song = resolve_output_song_dir(song_dir, source_dir, output_dir)
            rec = dest_recipe_index.get((str(out_song), track))
            meta_row = _stem_metadata_row(row, stems_df)
            try:
                category = listening_category_from_stem_row(meta_row)
                spec = recipe.spec_for_category(category)
            except (KeyError, TypeError, ValueError):
                continue
            backend = str(rec["backend"]) if rec and rec.get("backend") else "fluidsynth"
            if recorded_realify_fingerprint(rec) == desired_realify_fingerprint(
                spec, backend,
            ):
                continue
        source_stem_path = stem_path(song_dir, track, audio_format)
        if not stem_is_valid(source_stem_path):
            continue

        # DDSP: copy already-realified donor stems for soundfont fallbacks (skip SA3).
        if ddsp_mode and output_root and donor_mode:
            route = routing_index.get((str(song_dir), track), {})
            source_label = route.get("source")
            backend = route.get("backend")
            reuse = is_reused_source(source_label) or backend == "soundfont"
            if reuse:
                song_rel = song_rel_under_data(song_dir, source_dir)
                donor = donor_mode_from_source(source_label) or donor_mode
                donor_stem = donor_realify_stem_path(
                    output_root, donor, song_rel, track, audio_format,
                )
                if stem_is_valid(donor_stem):
                    copy_stem(donor_stem, out_path)
                    out_song = resolve_output_song_dir(song_dir, source_dir, output_dir)
                    original_path_updates.append(
                        (str(out_song), track, str(donor_stem.resolve()))
                    )
                    continue
                raise RuntimeError(
                    f"Missing donor realify stem for DDSP fallback: {donor_stem}\n"
                    f"Generate the donor realify ablation first:\n"
                    f"  uv run python -m synthesis.synthesize --render-mode {donor} --realify"
                )

        if presets is not None:
            meta_row = _stem_metadata_row(row, stems_df)
            if category_allowlist is not None:
                from synthesis.recipe import listening_category_from_stem_row

                category = listening_category_from_stem_row(meta_row)
                if category not in category_allowlist:
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    copy_stem(source_stem_path, out_path)
                    out_song = resolve_output_song_dir(song_dir, source_dir, output_dir)
                    original_path_updates.append(
                        (str(out_song), track, str(source_stem_path.resolve()))
                    )
                    continue
            preset = select_preset(presets, meta_row)
            if not realify_enabled(preset):
                out_path.parent.mkdir(parents=True, exist_ok=True)
                copy_stem(source_stem_path, out_path)
                out_song = resolve_output_song_dir(song_dir, source_dir, output_dir)
                original_path_updates.append(
                    (str(out_song), track, str(source_stem_path.resolve()))
                )
                continue
        tasks.append({
            "row": row.to_dict(),
            "out_path": str(out_path),
            "stem_path": str(source_stem_path),
            "duration": stem_duration_seconds(source_stem_path),
            "n_samples": stem_n_samples(source_stem_path),
            "audio_format": audio_format,
            "seed": stem_seed(sample_seed, str(song_dir), track),
        })

    if original_path_updates:
        _update_routing_original_paths(output_dir, original_path_updates)
    return tasks


def _update_routing_original_paths(
    output_dir: Path,
    updates: list[tuple[str, int, str]],
) -> None:
    """Set ``original_path`` on copied stems in the realify-tree ``ddsp_routing.csv``."""
    from synthesis.ddsp.config import DDSP_ROUTING_FILE_NAME

    routing_path = Path(output_dir) / DDSP_ROUTING_FILE_NAME
    if not routing_path.is_file() or not updates:
        return
    df = pd.read_csv(routing_path)
    if "original_path" not in df.columns:
        df["original_path"] = pd.NA
    for song_path, track, original in updates:
        mask = (df["path"].astype(str) == str(song_path)) & (df["track"].astype(int) == int(track))
        df.loc[mask, "original_path"] = original
    df.to_csv(routing_path, index=False)


def _shutdown_pool(pool) -> None:
    """Gracefully shut down a multiprocessing pool.

    Avoid ``with pool`` — Pool.__exit__ calls terminate(), which force-kills CUDA
    workers and triggers resource_tracker semaphore leak warnings on shutdown.
    """
    pool.close()
    pool.join()


def _teardown_gpu_realify_worker() -> None:
    global _REALIFY_MODEL, _REALIFY_PRESETS

    _REALIFY_MODEL = None
    _REALIFY_PRESETS = None
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        pass


def _init_gpu_realify_worker(
    model_name: str,
    presets_filepath: str,
    batch_size: int,
    silence_enforce: bool,
    content_fidelity_enforce: bool,
    audio_format: str,
    backend: str,
    progress_queue=None,
):
    global _REALIFY_MODEL, _REALIFY_PRESETS, _REALIFY_BATCH_SIZE, _REALIFY_WORKER_CONFIG
    global _REALIFY_PROGRESS_QUEUE

    configure_sa3_env()
    _REALIFY_PROGRESS_QUEUE = progress_queue
    _REALIFY_WORKER_CONFIG = {
        "model_name": model_name,
        "presets_filepath": presets_filepath,
        "batch_size": batch_size,
        "silence_enforce": silence_enforce,
        "content_fidelity_enforce": content_fidelity_enforce,
        "audio_format": audio_format,
        "backend": backend,
    }
    _REALIFY_BATCH_SIZE = batch_size
    _REALIFY_MODEL = None
    _REALIFY_PRESETS = None


def _ensure_gpu_realify_worker(device_id: int) -> None:
    global _REALIFY_MODEL, _REALIFY_PRESETS, _REALIFY_BATCH_SIZE

    if _REALIFY_MODEL is not None:
        return
    if _REALIFY_WORKER_CONFIG is None:
        raise RuntimeError("GPU realify worker config was not initialized")

    import torch

    cfg = _REALIFY_WORKER_CONFIG
    torch.cuda.set_device(device_id)
    print(
        f"Realify worker: loading SA3 {cfg['model_name']} on cuda:{device_id}...",
        flush=True,
    )
    _REALIFY_BATCH_SIZE = cfg["batch_size"]
    _REALIFY_PRESETS = load_presets(Path(cfg["presets_filepath"]))
    _REALIFY_MODEL = load_realify_model(
        cfg["model_name"],
        backend=cfg.get("backend", REALIFY_BACKEND),
        device_index=device_id,
    )
    print(f"Realify worker: cuda:{device_id} ready", flush=True)


def _init_cpu_realify_worker(
    model_name: str,
    presets_filepath: str,
    batch_size: int,
    silence_enforce: bool,
    content_fidelity_enforce: bool,
    audio_format: str,
    backend: str,
    progress_queue=None,
):
    global _REALIFY_MODEL, _REALIFY_PRESETS, _REALIFY_BATCH_SIZE, _REALIFY_WORKER_CONFIG
    global _REALIFY_PROGRESS_QUEUE

    configure_sa3_env()
    _REALIFY_PROGRESS_QUEUE = progress_queue
    _REALIFY_WORKER_CONFIG = {
        "model_name": model_name,
        "presets_filepath": presets_filepath,
        "batch_size": batch_size,
        "silence_enforce": silence_enforce,
        "content_fidelity_enforce": content_fidelity_enforce,
        "audio_format": audio_format,
        "backend": backend,
    }
    _REALIFY_BATCH_SIZE = batch_size
    _REALIFY_PRESETS = load_presets(Path(presets_filepath))
    _REALIFY_MODEL = load_realify_model(model_name, backend=backend)


def _realify_gpu_worker_shard(args: tuple[int, list[dict], int]) -> int:
    device_id, shard, batch_size = args
    if not shard:
        return 0
    try:
        _ensure_gpu_realify_worker(device_id)
        process_realify_tasks(
            shard,
            model=_REALIFY_MODEL,
            presets=_REALIFY_PRESETS,
            audio_format=_REALIFY_WORKER_CONFIG["audio_format"],
            batch_size=batch_size,
            desc="Realifying stems",
            show_progress=False,
            silence_enforce=_REALIFY_WORKER_CONFIG["silence_enforce"],
            content_fidelity_enforce=_REALIFY_WORKER_CONFIG["content_fidelity_enforce"],
        )
        return len(shard)
    finally:
        _teardown_gpu_realify_worker()


def _realify_worker_shard(shard: list[dict]) -> int:
    if not shard:
        return 0
    process_realify_tasks(
        shard,
        model=_REALIFY_MODEL,
        presets=_REALIFY_PRESETS,
        audio_format=_REALIFY_WORKER_CONFIG["audio_format"],
        batch_size=_REALIFY_BATCH_SIZE,
        desc="Realifying stems",
        show_progress=False,
        silence_enforce=_REALIFY_WORKER_CONFIG["silence_enforce"],
        content_fidelity_enforce=_REALIFY_WORKER_CONFIG["content_fidelity_enforce"],
    )
    return len(shard)


def _run_realify_gpu(
    tasks: list[dict],
    *,
    model: str,
    presets_filepath: Path,
    batch_size: int,
    audio_format: str,
    silence_enforce: bool = REALIFY_SILENCE_ENFORCE,
    content_fidelity_enforce: bool = REALIFY_CONTENT_FIDELITY_ENFORCE,
    backend: str = REALIFY_BACKEND,
) -> None:
    if content_fidelity_enforce:
        batch_size = 1
    if backend == "trt" and batch_size != 1:
        batch_size = 1
    gpu_indices = select_realify_gpu_indices()
    if not gpu_indices:
        raise RuntimeError(
            "No GPU has enough free VRAM for realify. "
            f"Need at least {REALIFY_MIN_GPU_FREE_GB:.1f} GiB free per worker. "
            f"Visible devices: {describe_visible_gpus()}. "
            "Free memory on busy GPUs, set CUDA_VISIBLE_DEVICES to idle devices "
            "(e.g. CUDA_VISIBLE_DEVICES=0,3), or use -m small-music for CPU realify."
        )

    presets = load_presets(presets_filepath)
    tasks = sort_realify_tasks_for_batching(tasks, presets)
    n_workers = min(len(gpu_indices), len(tasks))

    worker_batch_sizes: list[int] = []
    for device_id in gpu_indices[:n_workers]:
        free_gb, total_gb, _name = gpu_memory_snapshot(device_id)
        worker_batch_sizes.append(
            resolve_realify_batch_size(
                batch_size,
                free_gb=free_gb,
                total_gb=total_gb,
                model=model,
            )
        )

    if n_workers == 1:
        configure_sa3_env()
        import torch

        device_index = gpu_indices[0]
        torch.cuda.set_device(device_index)
        free_gb, total_gb, name = gpu_memory_snapshot(device_index)
        resolved_bs = worker_batch_sizes[0]
        print(
            f"Realify: loading SA3 {model} on GPU {device_index} "
            f"({name}, {free_gb:.1f}/{total_gb:.1f} GiB free); "
            f"batch_size={resolved_bs}"
            + (" (auto)" if batch_size is None or int(batch_size) <= 0 else "")
        )
        sa3_model = load_realify_model(
            model,
            backend=backend,
            device_index=device_index,
        )
        process_realify_tasks(
            tasks,
            model=sa3_model,
            presets=presets,
            audio_format=audio_format,
            batch_size=resolved_bs,
            desc="Realifying stems (GPU)",
            silence_enforce=silence_enforce,
            content_fidelity_enforce=content_fidelity_enforce,
            backend=backend,
        )
        return

    device_labels = []
    for device_id, bs in zip(gpu_indices[:n_workers], worker_batch_sizes):
        free_gb, total_gb, name = gpu_memory_snapshot(device_id)
        device_labels.append(
            f"GPU {device_id} ({name}, {free_gb:.1f}/{total_gb:.1f} GiB free, "
            f"batch={bs})"
        )
    print(
        f"Realify: loading SA3 {model} on {n_workers} GPU worker(s): "
        + "; ".join(device_labels)
    )
    shards = shard_tasks_contiguous(tasks, n_workers)
    shard_sizes = ", ".join(str(len(s)) for s in shards)
    print(
        f"Realify: {n_workers} workers × [{shard_sizes}] stems "
        "(category-sorted contiguous shards); "
        "progress updates as stems complete. "
        "Model load in workers is silent for a few minutes first.",
        flush=True,
    )

    ctx = multiprocessing.get_context("spawn")
    manager = ctx.Manager()
    progress_queue = manager.Queue()
    shard_args = [
        (gpu_indices[i], shards[i], worker_batch_sizes[i])
        for i in range(n_workers)
    ]
    # initargs batch_size is only a placeholder; each shard carries its own.
    pool = ctx.Pool(
        processes=n_workers,
        initializer=_init_gpu_realify_worker,
        initargs=(
            model,
            str(presets_filepath),
            max(worker_batch_sizes),
            silence_enforce,
            content_fidelity_enforce,
            audio_format,
            backend,
            progress_queue,
        ),
    )
    global _REALIFY_PROGRESS_QUEUE
    _REALIFY_PROGRESS_QUEUE = progress_queue
    try:
        desc = f"Realifying stems ({n_workers} GPUs)"
        _run_pool_with_stem_progress(
            pool,
            shard_args,
            total_tasks=len(tasks),
            desc=desc,
            worker_fn=_realify_gpu_worker_shard,
        )
    finally:
        _REALIFY_PROGRESS_QUEUE = None
        _shutdown_pool(pool)


def _run_realify_cpu(
    tasks: list[dict],
    *,
    model: str,
    presets_filepath: Path,
    jobs: int,
    batch_size: int,
    audio_format: str,
    silence_enforce: bool = REALIFY_SILENCE_ENFORCE,
    content_fidelity_enforce: bool = REALIFY_CONTENT_FIDELITY_ENFORCE,
    backend: str = REALIFY_BACKEND,
) -> None:
    if backend == "trt":
        raise RuntimeError("TensorRT backend requires a CUDA GPU.")
    if content_fidelity_enforce:
        batch_size = 1
    elif batch_size is None or int(batch_size) <= 0:
        # CPU path: no VRAM signal; keep batching modest.
        batch_size = 1 if model == "medium" else 2
    presets = load_presets(presets_filepath)
    tasks = sort_realify_tasks_for_batching(tasks, presets)
    n_workers = min(max(jobs, 1), len(tasks))

    if n_workers == 1:
        _init_cpu_realify_worker(
            model,
            str(presets_filepath),
            batch_size,
            silence_enforce,
            content_fidelity_enforce,
            audio_format,
            backend,
        )
        process_realify_tasks(
            tasks,
            model=_REALIFY_MODEL,
            presets=_REALIFY_PRESETS,
            audio_format=audio_format,
            batch_size=batch_size,
            desc="Realifying stems (CPU)",
            silence_enforce=silence_enforce,
            backend=backend,
        )
        return

    shards = shard_tasks_contiguous(tasks, n_workers)
    ctx = multiprocessing.get_context("spawn")
    manager = ctx.Manager()
    progress_queue = manager.Queue()
    pool = ctx.Pool(
        processes=n_workers,
        initializer=_init_cpu_realify_worker,
        initargs=(
            model,
            str(presets_filepath),
            batch_size,
            silence_enforce,
            content_fidelity_enforce,
            audio_format,
            backend,
            progress_queue,
        ),
    )
    global _REALIFY_PROGRESS_QUEUE
    _REALIFY_PROGRESS_QUEUE = progress_queue
    try:
        _run_pool_with_stem_progress(
            pool,
            shards,
            total_tasks=len(tasks),
            desc=f"Realifying stems ({n_workers} CPU workers)",
            worker_fn=_realify_worker_shard,
        )
    finally:
        _REALIFY_PROGRESS_QUEUE = None
        _shutdown_pool(pool)


def run_realify(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    model: str = "medium",
    limit: int | None = None,
    jobs: int = 1,
    batch_size: int = REALIFY_BATCH_SIZE,
    presets_filepath: Path | None = None,
    audio_format: str = DEFAULT_AUDIO_FORMAT,
    sample_seed: int = ABLATION_SAMPLE_SEED,
    reset: bool = False,
    silence_enforce: bool = REALIFY_SILENCE_ENFORCE,
    content_fidelity_enforce: bool = REALIFY_CONTENT_FIDELITY_ENFORCE,
    backend: str = REALIFY_BACKEND,
    output_root: str | None = None,
    render_mode: str | None = None,
    category_allowlist: set[str] | frozenset[str] | None = None,
    recipe=None,
    allowed_song_ids: set[str] | frozenset[str] | None = None,
):
    """Realify stems on visible GPU(s) or CPU (small-music only)."""
    configure_sa3_env()
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    presets_filepath = presets_filepath or DEFAULT_PRESETS_FILE

    if reset:
        if source_dir.resolve() == output_dir.resolve():
            print(
                "Skipping realify --reset (in-place would delete source stems).",
                flush=True,
            )
        else:
            reset_realify_output(output_dir)

    if batch_size is not None and int(batch_size) < 0:
        raise ValueError(f"batch_size must be >= 0 (0=auto), got {batch_size}")
    if content_fidelity_enforce:
        batch_size = 1

    if output_dir != source_dir:
        copy_metadata_tables(source_dir, output_dir)

    presets = load_presets(presets_filepath)
    captions = generate_captions(source_dir, seed=sample_seed, presets=presets)
    if allowed_song_ids is not None:
        from synthesis.shard import song_id_from_synthesis_path

        paths = captions["path"].astype(str)
        keep = paths.map(
            lambda p: song_id_from_synthesis_path(p) in allowed_song_ids
        )
        captions = captions.loc[keep].reset_index(drop=True)
    if limit:
        captions = captions.head(limit)

    tasks = build_realify_tasks(
        captions,
        source_dir,
        output_dir,
        audio_format,
        sample_seed=sample_seed,
        presets=presets,
        output_root=output_root,
        render_mode=render_mode,
        category_allowlist=category_allowlist,
        recipe=recipe,
        reset=reset,
    )
    use_gpu = (backend == "trt" or realify_uses_gpu(model)) if tasks else False
    log_realify_plan(
        source_dir=source_dir,
        output_dir=output_dir,
        model=model,
        n_tasks=len(tasks),
        n_captions=len(captions),
        use_gpu=use_gpu,
        batch_size=batch_size,
    )
    if not tasks:
        if recipe is not None:
            from synthesis.recipe import sync_realify_sidecar

            sync_realify_sidecar(source_dir, output_dir, recipe)
        return

    if use_gpu:
        _run_realify_gpu(
            tasks,
            model=model,
            presets_filepath=presets_filepath,
            batch_size=batch_size,
            audio_format=audio_format,
            silence_enforce=silence_enforce,
            content_fidelity_enforce=content_fidelity_enforce,
            backend=backend,
        )
    else:
        _run_realify_cpu(
            tasks,
            model=model,
            presets_filepath=presets_filepath,
            jobs=jobs,
            batch_size=batch_size,
            audio_format=audio_format,
            silence_enforce=silence_enforce,
            content_fidelity_enforce=content_fidelity_enforce,
            backend=backend,
        )
    if recipe is not None:
        from synthesis.recipe import sync_realify_sidecar

        sync_realify_sidecar(source_dir, output_dir, recipe)


def parse_args(args=None, namespace=None):
    import multiprocessing

    parser = argparse.ArgumentParser(description="Realify stems with Stable Audio 3.")
    parser.add_argument("--source-dir", default=None, type=str)
    parser.add_argument("--output-dir", default=None, type=str)
    parser.add_argument("-m", "--model", default="medium", choices=["small-music", "medium"])
    parser.add_argument("--limit", default=None, type=int)
    parser.add_argument(
        "-j",
        "--jobs",
        "--workers",
        default=int(multiprocessing.cpu_count() / 4),
        type=int,
        help="CPU workers for CPU realify (small-music).",
    )
    parser.add_argument(
        "--realify-batch-size",
        default=REALIFY_BATCH_SIZE,
        type=int,
        help=(
            "SA3 stems per GPU forward pass. 0=auto from each GPU's free/total VRAM "
            "(default: REALIFY_BATCH_SIZE in shared/config.py)."
        ),
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the realify output directory and re-realify all stems.",
    )
    parser.add_argument(
        "--flac",
        action="store_true",
        help="Read/write FLAC stems instead of the default MP3.",
    )
    parser.add_argument(
        "--no-silence-enforce",
        action="store_true",
        help="Disable post-SA3 silence enforcement (reference vs realified energy gating).",
    )
    parser.add_argument(
        "--content-fidelity-enforce",
        action="store_true",
        help="Enable onset-based content fidelity gate with init_noise_level backoff.",
    )
    parser.add_argument(
        "--no-content-fidelity-enforce",
        action="store_true",
        help="Disable content fidelity gate even if REALIFY_CONTENT_FIDELITY_ENFORCE is set.",
    )
    parser.add_argument(
        "--backend",
        default=REALIFY_BACKEND,
        choices=["pytorch", "trt"],
        help="SA3 inference backend (trt requires TensorRT engines installed).",
    )
    return parser.parse_args(args=args, namespace=namespace)


def main():
    from synthesis.mix import print_mix_hint

    args = parse_args()
    source_dir = args.source_dir or full_stems_dir(OUTPUT_DIR)
    output_dir = args.output_dir or source_dir
    from synthesis.audio import synthesis_audio_format

    content_fidelity_enforce = REALIFY_CONTENT_FIDELITY_ENFORCE
    if args.content_fidelity_enforce:
        content_fidelity_enforce = True
    if args.no_content_fidelity_enforce:
        content_fidelity_enforce = False

    audio_format = synthesis_audio_format(args.flac)
    run_realify(
        source_dir,
        output_dir,
        model=args.model,
        limit=args.limit,
        jobs=args.jobs,
        batch_size=args.realify_batch_size,
        audio_format=audio_format,
        reset=args.reset,
        silence_enforce=not args.no_silence_enforce,
        content_fidelity_enforce=content_fidelity_enforce,
        backend=args.backend,
    )
    print_mix_hint(output_dir, jobs=args.jobs, flac=bool(args.flac))


if __name__ == "__main__":
    main()
