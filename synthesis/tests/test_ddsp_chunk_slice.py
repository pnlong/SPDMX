"""Tests for MIDI-DDSP chunk slicing edge cases."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from synthesis.ddsp.env import DdspEnvError, ddsp_python_executable
from synthesis.ddsp.chunking import plan_chunk_spans


def _ddsp_env_available() -> bool:
    try:
        ddsp_python_executable()
        return True
    except DdspEnvError:
        return False


pytestmark = pytest.mark.skipif(
    not _ddsp_env_available(),
    reason="Neural DDSP TF venv not configured (see SETUP.md Track C)",
)


def test_slice_skips_sub_frame_edge_note():
    """Notes clipped to <4 ms at chunk boundaries must not crash synthesis."""
    pretty_midi = pytest.importorskip("pretty_midi")
    from synthesis.ddsp.worker import _slice_midi_time_window

    with tempfile.TemporaryDirectory() as tmp:
        stem = Path(tmp) / "stem.mid"
        # One note ending exactly at t=12s — chunk [100,112] clips it to ~0.11 ms.
        pm = pretty_midi.PrettyMIDI()
        inst = pretty_midi.Instrument(program=65, name="tenor sax")
        inst.notes.append(
            pretty_midi.Note(velocity=80, pitch=60, start=0.0, end=12.0)
        )
        pm.instruments.append(inst)
        pm.write(str(stem))

        sr = 16000
        total_samples = int(round(pm.get_end_time() * sr))
        spans = plan_chunk_spans(total_samples, int(12 * sr), int(2 * sr))
        # Chunk 10 is [100s, 112s] for this ~234s file.
        start_sec, end_sec = spans[10][0] / sr, spans[10][1] / sr
        chunk = Path(tmp) / "chunk.mid"
        has_notes = _slice_midi_time_window(stem, start_sec, end_sec, chunk)
        assert has_notes is False

        check = pretty_midi.PrettyMIDI(str(chunk))
        assert not check.instruments or not any(i.notes for i in check.instruments)


def test_slice_keeps_real_notes():
    pretty_midi = pytest.importorskip("pretty_midi")
    from synthesis.ddsp.worker import _slice_midi_time_window

    with tempfile.TemporaryDirectory() as tmp:
        stem = Path(tmp) / "stem.mid"
        pm = pretty_midi.PrettyMIDI()
        inst = pretty_midi.Instrument(program=65)
        inst.notes.append(
            pretty_midi.Note(velocity=80, pitch=60, start=1.0, end=3.0)
        )
        pm.instruments.append(inst)
        pm.write(str(stem))

        chunk = Path(tmp) / "chunk.mid"
        assert _slice_midi_time_window(stem, 0.5, 4.0, chunk) is True
        check = pretty_midi.PrettyMIDI(str(chunk))
        assert len(check.instruments) == 1
        assert len(check.instruments[0].notes) == 1
