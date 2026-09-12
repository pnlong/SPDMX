"""Audio synthesis helpers using fluidsynth."""

from __future__ import annotations

import contextlib
import os
import subprocess
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
import torch

from shared.config import (
    DEFAULT_AUDIO_FORMAT,
    FLAC_AUDIO_FORMAT,
    FLAC_SUBTYPE,
    GAIN,
    MAX_N_SAMPLES_IN_STEM,
    MIXTURE_PEAK_LIMIT,
    SAMPLE_RATE,
    STEM_CHANNELS,
    TARGET_LOUDNESS_LUFS,
)

SUPPORTED_AUDIO_FORMATS = frozenset({DEFAULT_AUDIO_FORMAT, FLAC_AUDIO_FORMAT})

# fluidsynth raw output: stereo int16 = 4 bytes per frame
_MAX_RAW_PCM_BYTES = MAX_N_SAMPLES_IN_STEM * 4


@contextlib.contextmanager
def _suppress_native_stderr():
    """Silence C-library stderr (e.g. libmpg123 one-frame MP3 warnings)."""
    try:
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        saved_fd = os.dup(2)
    except OSError:
        yield
        return
    try:
        os.dup2(devnull_fd, 2)
        yield
    finally:
        try:
            os.dup2(saved_fd, 2)
        finally:
            os.close(saved_fd)
            os.close(devnull_fd)


def truncate_waveform(
    waveform: torch.Tensor,
    max_samples: int = MAX_N_SAMPLES_IN_STEM,
) -> torch.Tensor:
    if waveform.shape[-1] <= max_samples:
        return waveform
    return waveform[:, :max_samples]


def get_waveform_tensor(
    midi_path: str,
    soundfont_filepath: str,
    *,
    fx_profile: str | None = None,
) -> torch.Tensor:
    """Synthesize a MIDI file to a float waveform tensor (channels, samples).

    fluidsynth output is capped at MAX_N_SAMPLES_IN_STEM so pathological MIDIs
    (missing note-offs, extreme length) cannot allocate multi-gigabyte buffers.
    """
    from synthesis.fx import apply_post_fx, fluidsynth_fx_args

    proc = subprocess.Popen(
        args=[
            "fluidsynth",
            "-T", "raw",
            "-F-", "-r", str(SAMPLE_RATE), "-g", str(GAIN),
            *fluidsynth_fx_args(fx_profile),
            "-i", soundfont_filepath,
            midi_path,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    assert proc.stdout is not None
    try:
        raw = proc.stdout.read(_MAX_RAW_PCM_BYTES)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    if len(raw) < 4:
        return torch.zeros((STEM_CHANNELS, 0))

    frame_bytes = 4
    n_frames = len(raw) // frame_bytes
    stereo = np.frombuffer(raw[: n_frames * frame_bytes], dtype=np.int16).reshape(-1, 2)
    scale = np.float32(1.0 / np.iinfo(np.int16).max)
    waveform = torch.from_numpy((stereo.astype(np.float32) * scale).T.copy())
    waveform = ensure_stem_channels(truncate_waveform(waveform))
    return apply_post_fx(waveform, fx_profile)


def to_mono_numpy(waveform: torch.Tensor) -> np.ndarray:
    """Convert waveform tensor to mono numpy array (samples,)."""
    audio = waveform.detach().cpu().numpy()
    if audio.ndim == 3:
        audio = audio[0]
    if audio.ndim == 2:
        if audio.shape[0] == 1:
            return audio[0]
        return audio.mean(axis=0)
    return audio


def ensure_stem_channels(
    waveform: torch.Tensor,
    channels: int | None = None,
) -> torch.Tensor:
    """Normalize a waveform tensor to (channels, samples)."""
    target = STEM_CHANNELS if channels is None else channels
    if target not in (1, 2):
        raise ValueError(f"unsupported stem channel count: {target}")

    audio = waveform.detach().cpu()
    if audio.ndim == 3:
        audio = audio[0]
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)

    current = audio.shape[0]
    if current == target:
        return audio.to(torch.float32)
    if target == 1:
        return audio.mean(dim=0, keepdim=True).to(torch.float32)
    if current == 1:
        return audio.repeat(2, 1).to(torch.float32)
    raise ValueError(
        f"cannot convert {current}-channel audio to {target} stem channels"
    )


def to_stem_numpy(
    waveform: torch.Tensor,
    channels: int | None = None,
) -> np.ndarray:
    """Return numpy audio for disk write: (samples,) mono or (samples, 2) stereo."""
    audio = ensure_stem_channels(waveform, channels)
    target = STEM_CHANNELS if channels is None else channels
    if target == 1:
        return audio[0].numpy()
    return audio.T.numpy()


def loudness_normalize(
    waveform: torch.Tensor,
    sample_rate: int = SAMPLE_RATE,
    target_lufs: float = TARGET_LOUDNESS_LUFS,
    peak_limit: float = MIXTURE_PEAK_LIMIT,
) -> torch.Tensor:
    """Normalize integrated loudness toward target LUFS without clipping peaks.

    pyloudnorm's ``normalize.loudness`` applies unlimited gain, which often pushes
    sparse MIDI stems above 1.0. We apply the LUFS gain only up to the peak limit.
    """
    audio = to_mono_numpy(waveform).astype(np.float64)
    if audio.size == 0 or np.max(np.abs(audio)) == 0:
        return waveform.type(torch.float)

    min_samples = int(sample_rate * 0.4)  # pyloudnorm default block size
    if audio.size < min_samples:
        return waveform.type(torch.float)

    meter = pyln.Meter(sample_rate)
    loudness = meter.integrated_loudness(audio)
    if np.isinf(loudness):
        return waveform.type(torch.float)

    peak = float(np.max(np.abs(audio)))
    loudness_gain = float(np.power(10.0, (target_lufs - loudness) / 20.0))
    peak_gain = peak_limit / peak if peak > 0 else 1.0
    gain = min(loudness_gain, peak_gain)
    stem = ensure_stem_channels(waveform)
    normalized = stem.numpy().astype(np.float64) * gain
    return torch.from_numpy(normalized.astype(np.float32))


def pad_and_loudness_normalize(
    waveforms: list[torch.Tensor],
    target_lufs: float = TARGET_LOUDNESS_LUFS,
) -> list[torch.Tensor]:
    """Loudness-normalize each stem, then zero-pad to equal length."""
    normalized = [loudness_normalize(w, target_lufs=target_lufs) for w in waveforms]
    max_length = max(w.shape[-1] for w in normalized)
    padded = []
    for waveform in normalized:
        padded.append(torch.nn.functional.pad(
            waveform,
            pad=(0, max_length - waveform.shape[-1]),
            mode="constant",
            value=0,
        ))
    return padded


def build_mixture(
    waveforms: list[torch.Tensor],
    peak_limit: float = MIXTURE_PEAK_LIMIT,
) -> tuple[torch.Tensor, float]:
    """Sum loudness-normalized stems and apply uniform anti-clip gain (Slakh-style).

    Returns ``(mixture, peak_gain)``. ``peak_gain`` is the same factor applied to the
    sum; callers should multiply released stems by it so ``sum(stems) == mixture``.
    """
    if not waveforms:
        raise ValueError("need at least one stem to build a mixture")
    max_length = max(w.shape[-1] for w in waveforms)
    aligned = [
        torch.nn.functional.pad(
            waveform,
            pad=(0, max_length - waveform.shape[-1]),
            mode="constant",
            value=0,
        )
        if waveform.shape[-1] < max_length
        else waveform
        for waveform in waveforms
    ]
    summed = torch.stack(aligned).sum(dim=0)
    audio = summed.numpy()
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    peak_gain = (peak_limit / peak) if peak > peak_limit else 1.0
    if peak_gain != 1.0:
        audio = audio * peak_gain
    return torch.from_numpy(audio.astype(np.float32)), float(peak_gain)


def apply_peak_gain(
    waveforms: list[torch.Tensor],
    peak_gain: float,
) -> list[torch.Tensor]:
    """Apply the mixture anti-clip factor to each stem (identity when gain is 1)."""
    if peak_gain == 1.0:
        return waveforms
    return [waveform * peak_gain for waveform in waveforms]


def synthesis_audio_format(use_flac: bool = False) -> str:
    return FLAC_AUDIO_FORMAT if use_flac else DEFAULT_AUDIO_FORMAT


def stem_filename(track: int, audio_format: str = DEFAULT_AUDIO_FORMAT) -> str:
    return f"{int(track)}.{audio_format}"


def legacy_stem_filename(track: int, audio_format: str = DEFAULT_AUDIO_FORMAT) -> str:
    """Previous on-disk name ``stem_{track}.{ext}``."""
    return f"stem_{int(track)}.{audio_format}"


def mixture_filename(audio_format: str = DEFAULT_AUDIO_FORMAT) -> str:
    return f"mixture.{audio_format}"


def stem_path(song_dir: Path, track: int, audio_format: str = DEFAULT_AUDIO_FORMAT) -> Path:
    canonical = Path(song_dir) / stem_filename(track, audio_format)
    if canonical.is_file():
        return canonical
    legacy = Path(song_dir) / legacy_stem_filename(track, audio_format)
    if legacy.is_file():
        return legacy
    return canonical


def mixture_path(song_dir: Path, audio_format: str = DEFAULT_AUDIO_FORMAT) -> Path:
    return song_dir / mixture_filename(audio_format)


def track_index_from_stem_name(name: str, audio_format: str) -> int | None:
    """Parse ``N.ext`` or ``stem_N.ext``; ignore mixture and other files."""
    suffix = f".{audio_format}"
    if not name.endswith(suffix):
        return None
    stem = name[: -len(suffix)]
    if stem.isdigit():
        return int(stem)
    if stem.startswith("stem_") and stem[len("stem_") :].isdigit():
        return int(stem[len("stem_") :])
    return None


def _stem_frame_count(path: Path) -> tuple[int, int]:
    with _suppress_native_stderr():
        info = sf.info(str(path))
    return int(info.frames), int(info.samplerate)


def stem_is_valid(path: Path) -> bool:
    """True when a stem file exists and is within MAX_N_SAMPLES_IN_STEM."""
    if not path.is_file():
        return False
    try:
        frames, _ = _stem_frame_count(path)
        return 0 < frames <= MAX_N_SAMPLES_IN_STEM
    except (RuntimeError, OSError, ValueError):
        return False


def stem_flac_is_valid(path: Path) -> bool:
    return stem_is_valid(path)


def flac_fully_decodes(path: Path) -> bool:
    """True when ``path`` exists and fully decodes to ≥1 sample (strict verify).

    Unlike ``stem_is_valid`` (header ``sf.info`` only), this reads the audio
    payload so truncated/corrupt FLAC bodies fail.
    """
    if not path.is_file():
        return False
    try:
        if path.stat().st_size <= 0:
            return False
        frames, sr = _stem_frame_count(path)
        if sr <= 0 or frames <= 0 or frames > MAX_N_SAMPLES_IN_STEM:
            return False
        with _suppress_native_stderr():
            audio, file_sr = sf.read(
                str(path), frames=frames, dtype="float32", always_2d=True,
            )
        if int(file_sr or 0) <= 0:
            return False
        return int(np.asarray(audio).size) > 0
    except (RuntimeError, OSError, ValueError):
        return False


def stem_duration_seconds(path: Path) -> float:
    """Duration in seconds, capped at MAX_N_SAMPLES_IN_STEM."""
    return stem_n_samples(path) / SAMPLE_RATE


def stem_n_samples(path: Path) -> int:
    """Sample count for a stem file, capped at MAX_N_SAMPLES_IN_STEM."""
    frames, _ = _stem_frame_count(path)
    return min(frames, MAX_N_SAMPLES_IN_STEM)


def load_stem(path: Path) -> torch.Tensor:
    """Load stem as float32 tensor (channels, samples), capped at MAX_N_SAMPLES_IN_STEM."""
    frames, _ = _stem_frame_count(path)
    frames = min(frames, MAX_N_SAMPLES_IN_STEM)
    if frames <= 0:
        return torch.zeros((STEM_CHANNELS, 0))
    with _suppress_native_stderr():
        audio, _ = sf.read(str(path), frames=frames, dtype="float32", always_2d=True)
    if audio.ndim == 1:
        audio = audio[:, np.newaxis]
    tensor = torch.from_numpy(np.asarray(audio.T, dtype=np.float32))
    return ensure_stem_channels(tensor)


def load_stem_flac(path: Path) -> torch.Tensor:
    return load_stem(path)


def write_mp3(waveform: torch.Tensor, path: Path) -> Path:
    import torchaudio

    path.parent.mkdir(parents=True, exist_ok=True)
    tensor = ensure_stem_channels(waveform)
    with _suppress_native_stderr():
        torchaudio.save(str(path), tensor, SAMPLE_RATE, format=DEFAULT_AUDIO_FORMAT)
    return path


def write_flac(waveform: torch.Tensor, path: Path) -> Path:
    """Write float32 waveform to FLAC using FLAC_SUBTYPE (PCM_16 on disk)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = to_stem_numpy(waveform).astype(np.float32)
    sf.write(str(path), audio, SAMPLE_RATE, format="FLAC", subtype=FLAC_SUBTYPE)
    return path


def list_stem_paths(
    song_dir: Path,
    audio_format: str = DEFAULT_AUDIO_FORMAT,
    *,
    require_valid: bool = True,
) -> list[Path]:
    """Sorted stem paths under ``song_dir`` (``N.ext`` or legacy ``stem_N.ext``)."""
    if not song_dir.is_dir():
        return []
    by_track: dict[int, Path] = {}
    for path in song_dir.iterdir():
        if not path.is_file():
            continue
        track = track_index_from_stem_name(path.name, audio_format)
        if track is None:
            continue
        current = by_track.get(track)
        if current is None or path.name == stem_filename(track, audio_format):
            by_track[track] = path
    paths = [by_track[t] for t in sorted(by_track)]
    if require_valid:
        return [p for p in paths if stem_is_valid(p)]
    return paths


def sum_stems_in_song_dir(
    song_dir: Path,
    audio_format: str = DEFAULT_AUDIO_FORMAT,
) -> torch.Tensor:
    """Load valid stems and return their sample-wise sum (pad to equal length).

    For ``*_summable`` trees the mix is exactly this sum (peak already baked in).
    Applies Slakh-style peak limiting only if the sum exceeds ``MIXTURE_PEAK_LIMIT``.
    """
    stem_paths = list_stem_paths(song_dir, audio_format, require_valid=True)
    if not stem_paths:
        raise FileNotFoundError(f"No valid stems under {song_dir}")
    waveforms = [load_stem(path) for path in stem_paths]
    mixture, _peak_gain = build_mixture(waveforms)
    return mixture


def encode_audio_bytes(
    waveform: torch.Tensor,
    audio_format: str = DEFAULT_AUDIO_FORMAT,
) -> bytes:
    """Encode a waveform to in-memory ``mp3`` / ``flac`` bytes."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"audio.{audio_format}"
        write_audio(waveform, path, audio_format)
        return path.read_bytes()


def write_audio(
    waveform: torch.Tensor,
    path: Path,
    audio_format: str = DEFAULT_AUDIO_FORMAT,
) -> Path:
    ext = path.suffix.lstrip(".").lower()
    if ext in SUPPORTED_AUDIO_FORMATS:
        audio_format = ext
    if audio_format == DEFAULT_AUDIO_FORMAT:
        return write_mp3(waveform, path)
    if audio_format == FLAC_AUDIO_FORMAT:
        return write_flac(waveform, path)
    raise ValueError(f"Unsupported audio format: {audio_format}")


def save_stem(
    waveform: torch.Tensor,
    song_dir: Path,
    track: int,
    audio_format: str = DEFAULT_AUDIO_FORMAT,
) -> Path:
    return write_audio(waveform, stem_path(song_dir, track, audio_format), audio_format)


def save_mixture(
    waveform: torch.Tensor,
    song_dir: Path,
    audio_format: str = DEFAULT_AUDIO_FORMAT,
) -> Path:
    return write_audio(waveform, mixture_path(song_dir, audio_format), audio_format)


def save_stem_flac(waveform: torch.Tensor, song_dir: Path, track: int) -> Path:
    return save_stem(waveform, song_dir, track, FLAC_AUDIO_FORMAT)


def save_mixture_flac(waveform: torch.Tensor, song_dir: Path) -> Path:
    return save_mixture(waveform, song_dir, FLAC_AUDIO_FORMAT)


def normalize_stems_for_sum(
    waveforms: list[torch.Tensor],
    song_dir: Path,
    track_indices: list[int],
    audio_format: str = DEFAULT_AUDIO_FORMAT,
    *,
    write_mixture: bool = False,
) -> tuple[list[torch.Tensor], float, Path | None]:
    """Peak-scale loudness-normalized stems so they remain linearly summable.

    Computes the Slakh-style anti-clip gain from the stem sum, applies that
    same factor to every stem, and writes stem files under ``song_dir``.
    Optionally also writes ``mixture.*`` (the mix is still just ``sum(stems)``).

    Returns ``(scaled_stems, peak_gain, mixture_path_or_none)``.
    """
    if len(waveforms) != len(track_indices):
        raise ValueError("waveforms and track_indices must have the same length")
    mixture, peak_gain = build_mixture(waveforms)
    scaled = apply_peak_gain(waveforms, peak_gain)
    for track, waveform in zip(track_indices, scaled):
        save_stem(waveform, song_dir, track, audio_format)
    mix_path = save_mixture(mixture, song_dir, audio_format) if write_mixture else None
    return scaled, peak_gain, mix_path


def write_stems_and_mixture(
    waveforms: list[torch.Tensor],
    song_dir: Path,
    track_indices: list[int],
    audio_format: str = DEFAULT_AUDIO_FORMAT,
    *,
    write_mixture: bool = False,
) -> tuple[list[torch.Tensor], Path | None]:
    """Normalize stems for summability; optionally also write ``mixture.*``."""
    scaled, _peak_gain, mix_path = normalize_stems_for_sum(
        waveforms,
        song_dir,
        track_indices,
        audio_format,
        write_mixture=write_mixture,
    )
    return scaled, mix_path


def normalize_stems_in_song_dir(
    song_dir: Path,
    track_indices: list[int],
    audio_format: str = DEFAULT_AUDIO_FORMAT,
    *,
    dest_song_dir: Path | None = None,
    write_mixture: bool = False,
    velocity_scales: dict[int, float] | None = None,
) -> float | None:
    """LUFS-normalize, apply velocity dynamics, peak-scale stems for summability.

    Loads from ``song_dir``, writes scaled stems to ``dest_song_dir`` (default:
    ``song_dir``, i.e. overwrite in place). Optionally writes ``mixture.*``.

    Pipeline: LUFS → × velocity_scale[track] → sum → shared peak_gain → write.
    Returns the anti-clip ``peak_gain``, or ``None`` if any stem is missing/invalid.

    Songs whose padded float32 footprint would exceed
    ``SPDMX_MIX_NORMALIZE_RAM_BYTES`` (default 2 GiB) use a disk-staging path so
    many long stems cannot pin tens of GB and wedge the mix worker pool.
    """
    from synthesis.velocity import apply_velocity_scales

    stem_paths = [stem_path(song_dir, track, audio_format) for track in track_indices]
    if not all(stem_is_valid(path) for path in stem_paths):
        return None
    dest = dest_song_dir if dest_song_dir is not None else song_dir
    dest.mkdir(parents=True, exist_ok=True)

    frame_counts = [stem_n_samples(path) for path in stem_paths]
    max_frames = max(frame_counts) if frame_counts else 0
    if max_frames <= 0:
        return None
    # In-memory path holds ~3× float32 copies of every padded stem.
    est_bytes = len(stem_paths) * max_frames * STEM_CHANNELS * 4 * 3
    if est_bytes <= _mix_normalize_ram_budget_bytes():
        waveforms = pad_and_loudness_normalize([load_stem(path) for path in stem_paths])
        waveforms = apply_velocity_scales(waveforms, track_indices, velocity_scales)
        _scaled, peak_gain, _mix_path = normalize_stems_for_sum(
            waveforms,
            dest,
            track_indices,
            audio_format,
            write_mixture=write_mixture,
        )
        return peak_gain

    print(
        f"mix normalize: large song ({len(stem_paths)} stems, "
        f"~{est_bytes / (1024 ** 3):.1f} GiB if buffered) — disk-staging under {dest}",
        flush=True,
    )
    return _normalize_stems_in_song_dir_staged(
        stem_paths,
        track_indices,
        dest,
        audio_format=audio_format,
        write_mixture=write_mixture,
        velocity_scales=velocity_scales,
        max_frames=max_frames,
    )


def _mix_normalize_ram_budget_bytes() -> int:
    import os

    raw = os.environ.get("SPDMX_MIX_NORMALIZE_RAM_BYTES", str(2 * 1024 ** 3))
    try:
        return max(64 * 1024 ** 2, int(raw))
    except ValueError:
        return 2 * 1024 ** 3


def _peak_of_sum_flac_files(
    paths: list[Path],
    *,
    max_frames: int,
    block: int = 262_144,
) -> float:
    """Max |sum| over time without loading every stem fully into RAM."""
    peak = 0.0
    for start in range(0, max_frames, block):
        n = min(block, max_frames - start)
        acc = np.zeros((n, STEM_CHANNELS), dtype=np.float64)
        for path in paths:
            with _suppress_native_stderr():
                audio, _ = sf.read(
                    str(path),
                    start=start,
                    frames=n,
                    dtype="float32",
                    always_2d=True,
                )
            if audio.shape[0] == 0:
                continue
            if audio.shape[1] == 1 and STEM_CHANNELS > 1:
                audio = np.repeat(audio, STEM_CHANNELS, axis=1)
            elif audio.shape[1] > STEM_CHANNELS:
                audio = audio[:, :STEM_CHANNELS]
            elif audio.shape[1] < STEM_CHANNELS:
                pad_ch = np.zeros((audio.shape[0], STEM_CHANNELS), dtype=np.float32)
                pad_ch[:, : audio.shape[1]] = audio
                audio = pad_ch
            if audio.shape[0] < n:
                padded = np.zeros((n, STEM_CHANNELS), dtype=np.float32)
                padded[: audio.shape[0]] = audio
                audio = padded
            acc += audio.astype(np.float64, copy=False)
        peak = max(peak, float(np.max(np.abs(acc))) if acc.size else 0.0)
    return peak


def _normalize_stems_in_song_dir_staged(
    stem_paths: list[Path],
    track_indices: list[int],
    dest: Path,
    *,
    audio_format: str,
    write_mixture: bool,
    velocity_scales: dict[int, float] | None,
    max_frames: int,
) -> float:
    """LUFS + velocity to staging FLACs, block-peak sum, then peak-scale to dest."""
    import shutil
    import tempfile

    from synthesis.velocity import apply_velocity_scales

    staging = Path(
        tempfile.mkdtemp(prefix=".mix_norm_", dir=str(dest)),
    )
    try:
        staged: list[Path] = []
        for path, track in zip(stem_paths, track_indices):
            wave = loudness_normalize(load_stem(path))
            wave = apply_velocity_scales([wave], [track], velocity_scales)[0]
            out = staging / f"{int(track)}.flac"
            write_flac(wave, out)
            staged.append(out)
            del wave

        peak = _peak_of_sum_flac_files(staged, max_frames=max_frames)
        peak_gain = (MIXTURE_PEAK_LIMIT / peak) if peak > MIXTURE_PEAK_LIMIT else 1.0

        mix_acc: np.ndarray | None = None
        if write_mixture:
            mix_acc = np.zeros((max_frames, STEM_CHANNELS), dtype=np.float64)

        for staged_path, track in zip(staged, track_indices):
            wave = load_stem(staged_path)
            if peak_gain != 1.0:
                wave = wave * peak_gain
            save_stem(wave, dest, track, audio_format)
            if mix_acc is not None:
                arr = to_stem_numpy(wave).astype(np.float64, copy=False)
                if arr.ndim == 1:
                    arr = arr[:, np.newaxis]
                n = min(arr.shape[0], max_frames)
                mix_acc[:n] += arr[:n]
            del wave

        if write_mixture and mix_acc is not None:
            mix = torch.from_numpy(mix_acc.T.astype(np.float32))
            save_mixture(mix, dest, audio_format)
        return float(peak_gain)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def write_mixture_from_waveforms(
    waveforms: list[torch.Tensor],
    song_dir: Path,
    audio_format: str = DEFAULT_AUDIO_FORMAT,
    *,
    track_indices: list[int] | None = None,
) -> Path:
    """Write ``mixture.*``. When ``track_indices`` is set, also rewrite stems."""
    if track_indices is None:
        mixture, _peak_gain = build_mixture(waveforms)
        return save_mixture(mixture, song_dir, audio_format)
    _scaled, mix_path = write_stems_and_mixture(
        waveforms, song_dir, track_indices, audio_format, write_mixture=True,
    )
    assert mix_path is not None
    return mix_path


def write_mixture_from_song_dir(
    song_dir: Path,
    track_indices: list[int],
    audio_format: str = DEFAULT_AUDIO_FORMAT,
) -> Path | None:
    """Normalize stems in place for summability (no mixture file written).

    Kept for callers that historically built a mixture; prefer
    ``normalize_stems_in_song_dir``. Returns ``song_dir`` on success, else None.
    """
    peak_gain = normalize_stems_in_song_dir(song_dir, track_indices, audio_format)
    return song_dir if peak_gain is not None else None


def stem_flac_path(song_dir: Path, track: int) -> Path:
    return stem_path(song_dir, track, FLAC_AUDIO_FORMAT)


def mixture_flac_path(song_dir: Path) -> Path:
    return mixture_path(song_dir, FLAC_AUDIO_FORMAT)


def song_is_complete(
    song_dir: Path,
    n_tracks: int,
    audio_format: str = DEFAULT_AUDIO_FORMAT,
    *,
    require_mixture: bool = False,
) -> bool:
    """Return True if all expected valid stems exist (and mixture, if required)."""
    if not song_dir.is_dir():
        return False
    stems_ok = all(
        stem_is_valid(stem_path(song_dir, j, audio_format)) for j in range(n_tracks)
    )
    if not stems_ok:
        return False
    if not require_mixture:
        return True
    return stem_is_valid(mixture_path(song_dir, audio_format))
