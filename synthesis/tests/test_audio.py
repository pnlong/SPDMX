"""Tests for mono downmix and BS.1770 loudness normalization."""

from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from shared.config import (
    DEFAULT_AUDIO_FORMAT,
    FLAC_AUDIO_FORMAT,
    FLAC_SUBTYPE,
    MAX_N_SAMPLES_IN_STEM,
    SAMPLE_RATE,
)
from synthesis.audio import (
    apply_peak_gain,
    build_mixture,
    ensure_stem_channels,
    load_stem,
    load_stem_flac,
    loudness_normalize,
    pad_and_loudness_normalize,
    save_stem,
    song_is_complete,
    stem_filename,
    stem_flac_is_valid,
    stem_path,
    to_mono_numpy,
    truncate_waveform,
    write_flac,
    write_mixture_from_song_dir,
    write_mp3,
    write_audio,
    write_stems_and_mixture,
)


def test_to_mono_from_stereo_tensor():
    stereo = torch.tensor([[1.0, -1.0, 1.0], [-1.0, 1.0, -1.0]])
    mono = to_mono_numpy(stereo)
    assert mono.shape == (3,)
    np.testing.assert_allclose(mono, [0.0, 0.0, 0.0])


def test_to_mono_from_batched_tensor():
    batched = torch.tensor([[[1.0, -1.0], [-1.0, 1.0]]])
    mono = to_mono_numpy(batched)
    assert mono.shape == (2,)
    np.testing.assert_allclose(mono, [0.0, 0.0])


def test_ensure_stem_channels_mono_downmix():
    stereo = torch.tensor([[1.0, -1.0, 1.0], [-1.0, 1.0, -1.0]])
    mono = ensure_stem_channels(stereo, channels=1)
    assert mono.shape == (1, 3)
    np.testing.assert_allclose(mono[0], [0.0, 0.0, 0.0])


def test_ensure_stem_channels_mono_upmix():
    mono = torch.tensor([[0.5, 1.0, 1.5]])
    stereo = ensure_stem_channels(mono, channels=2)
    assert stereo.shape == (2, 3)
    np.testing.assert_allclose(stereo[0], mono[0])
    np.testing.assert_allclose(stereo[1], mono[0])


def test_get_waveform_tensor_preserves_fluidsynth_stereo(monkeypatch):
    import synthesis.audio as audio_mod

    monkeypatch.setattr(audio_mod, "STEM_CHANNELS", 2)
    left = np.array([1000, 2000], dtype=np.int16)
    right = np.array([-1000, -2000], dtype=np.int16)
    raw = np.column_stack([left, right]).astype(np.int16).tobytes()

    class FakeStdout:
        def read(self, _n):
            return raw

    class FakeProc:
        stdout = FakeStdout()

        def terminate(self):
            return None

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(audio_mod.subprocess, "Popen", lambda **kwargs: FakeProc())
    waveform = audio_mod.get_waveform_tensor("song.mid", "font.sf2")
    assert waveform.shape == (2, 2)
    assert waveform[0, 0] != waveform[1, 0]
    np.testing.assert_allclose(waveform[0, 0], 1000 / np.iinfo(np.int16).max, rtol=1e-5)
    np.testing.assert_allclose(waveform[1, 0], -1000 / np.iinfo(np.int16).max, rtol=1e-5)


def test_get_waveform_tensor_downmixes_when_mono_configured(monkeypatch):
    import synthesis.audio as audio_mod

    monkeypatch.setattr(audio_mod, "STEM_CHANNELS", 1)
    stereo = np.array([[1000, -1000], [2000, -2000]], dtype=np.int16)
    raw = stereo.tobytes()

    class FakeStdout:
        def read(self, _n):
            return raw

    class FakeProc:
        stdout = FakeStdout()

        def terminate(self):
            return None

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(audio_mod.subprocess, "Popen", lambda **kwargs: FakeProc())
    waveform = audio_mod.get_waveform_tensor("song.mid", "font.sf2")
    assert waveform.shape == (1, 2)
    np.testing.assert_allclose(waveform[0, 0], 0.0, atol=1e-5)


def test_write_flac_stereo(tmp_path: Path, monkeypatch):
    import synthesis.audio as audio_mod

    captured = {}
    monkeypatch.setattr(audio_mod, "STEM_CHANNELS", 2)

    def fake_write(path, audio, sr, format, subtype):
        captured.update(shape=audio.shape)

    monkeypatch.setattr(audio_mod.sf, "write", fake_write)
    write_flac(
        torch.tensor([[0.5, 1.0], [0.25, 0.75]]),
        tmp_path / "stem_0.flac",
    )
    assert captured["shape"] == (2, 2)


def test_loudness_normalize_non_silent():
    t = torch.linspace(0, 1, SAMPLE_RATE)
    waveform = torch.sin(2 * np.pi * 440 * t).unsqueeze(0) * 0.01
    normalized = loudness_normalize(waveform)
    assert normalized.shape == waveform.shape
    assert normalized.abs().max() <= 1.0 + 1e-6


def test_loudness_normalize_caps_peak_for_quiet_sparse_stem():
    """Quiet stem with loud peaks: unlimited LUFS gain would clip; we peak-limit."""
    sr = SAMPLE_RATE
    audio = np.zeros(sr, dtype=np.float64)
    audio[1000] = 0.5
    audio[5000] = 0.3
    waveform = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
    normalized = loudness_normalize(waveform)
    assert normalized.abs().max().item() <= 1.0 + 1e-6
    assert normalized.abs().max().item() > 0.3


def test_pad_and_loudness_equal_length():
    sr = SAMPLE_RATE
    w1 = torch.randn(1, sr)
    w2 = torch.randn(1, sr * 2)
    padded = pad_and_loudness_normalize([w1, w2])
    assert padded[0].shape[-1] == padded[1].shape[-1] == sr * 2


def test_normalize_stems_staged_matches_in_memory(tmp_path: Path, monkeypatch):
    """Low RAM budget forces disk-staging path; peak gain matches in-memory path."""
    from synthesis.audio import normalize_stems_in_song_dir, load_stem, to_mono_numpy

    src = tmp_path / "raw"
    dest_mem = tmp_path / "audio_mem"
    dest_staged = tmp_path / "audio_staged"
    src.mkdir()
    sr = SAMPLE_RATE
    for track, amp in ((0, 0.4), (1, 0.5), (2, 0.3)):
        sf.write(
            str(src / f"{track}.flac"),
            np.full(max(sr // 5, int(0.5 * sr)), amp, np.float32),
            sr,
            format="FLAC",
        )

    monkeypatch.setenv("SPDMX_MIX_NORMALIZE_RAM_BYTES", str(64 * 1024 ** 3))  # in-memory
    gain_mem = normalize_stems_in_song_dir(
        src, [0, 1, 2], "flac", dest_song_dir=dest_mem, velocity_scales=None,
    )
    monkeypatch.setenv("SPDMX_MIX_NORMALIZE_RAM_BYTES", "1")  # force staged
    gain_staged = normalize_stems_in_song_dir(
        src, [0, 1, 2], "flac", dest_song_dir=dest_staged, velocity_scales=None,
    )
    assert gain_mem is not None and gain_staged is not None
    np.testing.assert_allclose(gain_mem, gain_staged, rtol=1e-4, atol=1e-4)
    for track in (0, 1, 2):
        a = to_mono_numpy(load_stem(dest_mem / f"{track}.flac"))
        b = to_mono_numpy(load_stem(dest_staged / f"{track}.flac"))
        np.testing.assert_allclose(a, b, rtol=1e-3, atol=1e-3)


def test_build_mixture_scales_when_clipping():
    w1 = torch.ones(1, 4) * 0.8
    w2 = torch.ones(1, 4) * 0.8
    mixture, peak_gain = build_mixture([w1, w2], peak_limit=1.0)
    assert mixture.abs().max().item() <= 1.0 + 1e-6
    np.testing.assert_allclose(peak_gain, 1.0 / 1.6, rtol=1e-5)
    np.testing.assert_allclose(to_mono_numpy(mixture), [1.0, 1.0, 1.0, 1.0], rtol=1e-5)


def test_build_mixture_single_stem():
    w = torch.linspace(0, 1, 100).unsqueeze(0)
    mixture, peak_gain = build_mixture([w])
    assert peak_gain == 1.0
    np.testing.assert_allclose(to_mono_numpy(mixture), to_mono_numpy(w))


def test_build_mixture_pads_mismatched_lengths():
    w1 = torch.ones(1, 5) * 0.4
    w2 = torch.ones(1, 4) * 0.4
    mixture, peak_gain = build_mixture([w1, w2])
    assert peak_gain == 1.0
    assert mixture.shape[-1] == 5
    np.testing.assert_allclose(to_mono_numpy(mixture), [0.8, 0.8, 0.8, 0.8, 0.4], rtol=1e-5)


def test_apply_peak_gain_identity():
    w = torch.ones(1, 4) * 0.5
    out = apply_peak_gain([w], 1.0)
    assert out[0] is w


def test_stems_sum_to_mixture_after_peak_scale():
    w1 = torch.ones(1, 4) * 0.8
    w2 = torch.ones(1, 4) * 0.8
    mixture, peak_gain = build_mixture([w1, w2], peak_limit=1.0)
    scaled = apply_peak_gain([w1, w2], peak_gain)
    summed = scaled[0] + scaled[1]
    np.testing.assert_allclose(to_mono_numpy(summed), to_mono_numpy(mixture), rtol=1e-5)


def test_write_stems_and_mixture_rewrites_peak_scaled_stems(tmp_path: Path):
    song_dir = tmp_path / "song"
    song_dir.mkdir()
    w1 = torch.ones(1, SAMPLE_RATE) * 0.8
    w2 = torch.ones(1, SAMPLE_RATE) * 0.8
    scaled, mix_path = write_stems_and_mixture(
        [w1, w2], song_dir, [0, 1], FLAC_AUDIO_FORMAT,
    )
    assert mix_path is None  # mixture not written by default
    assert not (song_dir / "mixture.flac").exists()
    stem0 = load_stem(song_dir / "0.flac")
    stem1 = load_stem(song_dir / "1.flac")
    summed = stem0 + stem1
    assert summed.abs().max().item() <= 1.0 + 1e-4
    np.testing.assert_allclose(
        to_mono_numpy(stem0), to_mono_numpy(scaled[0]), rtol=1e-4, atol=1e-4,
    )


def test_write_mixture_from_song_dir(tmp_path: Path):
    song_dir = tmp_path / "song"
    song_dir.mkdir()
    sr = 44100
    sf.write(str(song_dir / "stem_0.mp3"), np.full(sr, 0.5, dtype=np.float32), sr, format="MP3")
    sf.write(str(song_dir / "stem_1.mp3"), np.full(sr, 0.5, dtype=np.float32), sr, format="MP3")
    out = write_mixture_from_song_dir(song_dir, [0, 1])
    assert out is not None
    assert out == song_dir
    assert not (song_dir / "mixture.mp3").exists()


def test_write_mixture_from_song_dir_makes_stems_summable(tmp_path: Path):
    song_dir = tmp_path / "song"
    song_dir.mkdir()
    # Loud constant stems: after LUFS they still sum above 1.0 and need anti-clip.
    sr = SAMPLE_RATE
    sf.write(
        str(song_dir / "stem_0.flac"),
        np.full(sr, 0.9, dtype=np.float32),
        sr,
        format="FLAC",
    )
    sf.write(
        str(song_dir / "stem_1.flac"),
        np.full(sr, 0.9, dtype=np.float32),
        sr,
        format="FLAC",
    )
    out = write_mixture_from_song_dir(song_dir, [0, 1], FLAC_AUDIO_FORMAT)
    assert out is not None
    assert not (song_dir / "mixture.flac").exists()
    stem0 = load_stem(song_dir / "stem_0.flac")
    stem1 = load_stem(song_dir / "stem_1.flac")
    summed = stem0 + stem1
    assert summed.abs().max().item() <= 1.0 + 1e-5


def test_song_is_complete_requires_mixture(tmp_path: Path):
    song_dir = tmp_path / "song"
    song_dir.mkdir()
    sr = 44100
    sf.write(str(song_dir / "stem_0.mp3"), np.zeros(sr, dtype=np.float32), sr, format="MP3")
    assert song_is_complete(song_dir, 1)
    assert not song_is_complete(song_dir, 1, require_mixture=True)
    write_stems_and_mixture(
        [torch.zeros(1, sr)], song_dir, [0], write_mixture=True,
    )
    assert song_is_complete(song_dir, 1, require_mixture=True)


def test_loudness_normalize_silent_passthrough():
    silent = torch.zeros(1, SAMPLE_RATE)
    out = loudness_normalize(silent)
    assert out.shape == silent.shape


def test_truncate_waveform():
    w = torch.randn(1, MAX_N_SAMPLES_IN_STEM + 1000)
    out = truncate_waveform(w)
    assert out.shape[-1] == MAX_N_SAMPLES_IN_STEM
    short = torch.randn(1, 100)
    assert truncate_waveform(short).shape == short.shape


def test_stem_flac_is_valid_rejects_oversized(tmp_path: Path, monkeypatch):
    path = tmp_path / "stem_0.flac"
    path.write_bytes(b"not a real flac")
    assert not stem_flac_is_valid(path)

    import synthesis.audio as audio_mod

    class FakeInfo:
        frames = MAX_N_SAMPLES_IN_STEM + 1
        samplerate = SAMPLE_RATE

    monkeypatch.setattr(audio_mod.sf, "info", lambda _: FakeInfo())
    path.write_bytes(b"x" * 8)
    assert not stem_flac_is_valid(path)


def test_write_mp3(tmp_path: Path, monkeypatch):
    import synthesis.audio as audio_mod

    captured = {}

    def fake_save(path, tensor, sample_rate, format):
        captured.update(path=path, format=format, shape=tuple(tensor.shape))

    monkeypatch.setattr("torchaudio.save", fake_save)
    write_mp3(torch.ones(1, 100), tmp_path / "stem_0.mp3")
    assert captured["format"] == "mp3"
    assert captured["shape"] == (1, 100)


def test_write_audio_uses_path_extension_over_format_arg(tmp_path: Path, monkeypatch):
    import synthesis.audio as audio_mod

    captured = {}

    def fake_write(path, audio, sr, format, subtype):
        captured.update(format=format, path=path)

    monkeypatch.setattr(audio_mod.sf, "write", fake_write)
    write_audio(torch.ones(1, 100), tmp_path / "stem_0.flac", audio_format=FLAC_AUDIO_FORMAT)
    assert captured["format"] == "FLAC"
    assert str(captured["path"]).endswith(".flac")


def test_save_stem_mp3_uses_mp3_extension(tmp_path: Path, monkeypatch):
    import synthesis.audio as audio_mod

    monkeypatch.setattr("torchaudio.save", lambda *args, **kwargs: None)
    out = save_stem(torch.ones(1, 100), tmp_path, 0, DEFAULT_AUDIO_FORMAT)
    assert out.name == "0.mp3"


def test_stem_path_falls_back_to_legacy_stem_prefix(tmp_path: Path):
    legacy = tmp_path / "stem_0.flac"
    legacy.write_bytes(b"x")
    assert stem_path(tmp_path, 0, "flac") == legacy
    assert stem_filename(0, "flac") == "0.flac"
    numbered = tmp_path / "0.flac"
    numbered.write_bytes(b"y")
    assert stem_path(tmp_path, 0, "flac") == numbered


def test_write_flac_uses_pcm_16_subtype(tmp_path: Path, monkeypatch):
    import synthesis.audio as audio_mod

    captured = {}

    def fake_write(path, audio, sr, format, subtype):
        captured.update(path=path, subtype=subtype, dtype=audio.dtype)

    monkeypatch.setattr(audio_mod.sf, "write", fake_write)
    write_flac(torch.ones(1, 100), tmp_path / "stem_0.flac")
    assert captured["subtype"] == FLAC_SUBTYPE
    assert captured["dtype"] == np.float32


def test_load_stem_flac_caps_frames(tmp_path: Path, monkeypatch):
    import synthesis.audio as audio_mod

    class FakeInfo:
        frames = MAX_N_SAMPLES_IN_STEM + 5000
        samplerate = SAMPLE_RATE

    monkeypatch.setattr(audio_mod.sf, "info", lambda _: FakeInfo())

    def fake_read(path, frames, dtype, always_2d):
        assert frames == MAX_N_SAMPLES_IN_STEM
        return np.zeros(frames, dtype=np.float32), SAMPLE_RATE

    monkeypatch.setattr(audio_mod.sf, "read", fake_read)
    out = load_stem_flac(tmp_path / "stem_0.flac")
    assert out.shape == (1, MAX_N_SAMPLES_IN_STEM)
