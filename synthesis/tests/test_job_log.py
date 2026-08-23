"""Tests for synthesis.job_log."""

from __future__ import annotations

import tempfile
from pathlib import Path

from synthesis.job_log import SynthesisJobLog, close_job_log, get_job_log, open_job_log


def test_job_log_writes_and_registers_global():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "synthesis.midi_ddsp.log"
        log = open_job_log(path, synthesis_pass="midi_ddsp", workers=4)
        assert get_job_log() is log
        log.song_start(0, "/songs/foo")
        log.song_done(0, "/songs/foo", n_tracks=3)
        log.song_error(1, "/songs/bar", ValueError("boom"))
        close_job_log()
        assert get_job_log() is None
        text = path.read_text(encoding="utf-8")
        assert "song start" in text
        assert "song done" in text
        assert "song failed" in text
        assert "ValueError: boom" in text


def test_job_log_interrupted_message():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "job.log"
        log = SynthesisJobLog(path)
        log.interrupted(where="test")
        text = path.read_text(encoding="utf-8")
        assert "interrupted" in text
        assert "Ctrl+C" in text
        log.close()
