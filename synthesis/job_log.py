"""Append-only job log for long hybrid synthesis passes (multi-GPU DDSP, etc.)."""

from __future__ import annotations

import os
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_CURRENT: "SynthesisJobLog | None" = None
_LOCK = threading.Lock()


def job_log_enabled() -> bool:
    return os.environ.get("SPDMX_SYNTH_LOG", "1") != "0"


def get_job_log() -> "SynthesisJobLog | None":
    return _CURRENT


def open_job_log(path: str | Path, **meta: Any) -> "SynthesisJobLog":
    """Open (append) a pass log and register it as the process-global log."""
    global _CURRENT
    log = SynthesisJobLog(path, **meta)
    with _LOCK:
        _CURRENT = log
    return log


def close_job_log() -> None:
    global _CURRENT
    with _LOCK:
        if _CURRENT is not None:
            _CURRENT.close()
            _CURRENT = None


class SynthesisJobLog:
    """Thread-safe append log; also mirrors important lines to stdout."""

    def __init__(self, path: str | Path, **meta: Any):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "a", encoding="utf-8", buffering=1)
        self._lock = threading.Lock()
        self._tracks_done = 0
        self._songs_done = 0
        self._last_heartbeat = time.monotonic()
        pid = os.getpid()
        self.info(
            f"=== synthesis job log opened (pid={pid}) ===",
            **meta,
        )

    def _ts(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    def _write(self, level: str, message: str, *, echo: bool = True) -> None:
        line = f"[{self._ts()}] [{level}] {message}\n"
        with self._lock:
            self._file.write(line)
            self._file.flush()
        if echo:
            print(line.rstrip(), flush=True, file=sys.stdout)

    def info(self, message: str, **fields: Any) -> None:
        extra = ""
        if fields:
            parts = [f"{k}={fields[k]!r}" for k in sorted(fields)]
            extra = " " + " ".join(parts)
        self._write("INFO", message + extra)

    def warn(self, message: str) -> None:
        self._write("WARN", message)

    def error(self, message: str) -> None:
        self._write("ERROR", message)

    def song_start(self, song_index: int, path_output: str) -> None:
        self.info("song start", index=song_index, path=path_output)

    def song_done(
        self,
        song_index: int,
        path_output: str,
        *,
        n_tracks: int,
    ) -> None:
        self._songs_done += 1
        self._tracks_done += int(n_tracks)
        self.info(
            "song done",
            index=song_index,
            path=path_output,
            n_tracks=int(n_tracks),
            songs_total=self._songs_done,
            tracks_total=self._tracks_done,
        )
        self._maybe_heartbeat()

    def song_error(
        self,
        song_index: int,
        path_output: str,
        exc: BaseException,
    ) -> None:
        self._write(
            "ERROR",
            f"song failed index={song_index} path={path_output!r}: "
            f"{type(exc).__name__}: {exc}",
        )
        tb = traceback.format_exc()
        with self._lock:
            self._file.write(tb)
            if not tb.endswith("\n"):
                self._file.write("\n")
            self._file.flush()

    def interrupted(self, *, where: str) -> None:
        self.warn(
            f"job interrupted ({where}) — likely Ctrl+C / SIGINT in this terminal "
            f"(songs_done={self._songs_done}, tracks_done={self._tracks_done}). "
            f"Re-run the same command to resume; see {self.path}"
        )

    def fatal(self, exc: BaseException, *, where: str) -> None:
        self._write(
            "FATAL",
            f"job aborted at {where}: {type(exc).__name__}: {exc}",
        )
        tb = traceback.format_exc()
        with self._lock:
            self._file.write(tb)
            if not tb.endswith("\n"):
                self._file.write("\n")
            self._file.flush()

    def ddsp_failure(self, message: str, *, stderr_tail: str = "") -> None:
        self.error(message)
        if stderr_tail:
            with self._lock:
                self._file.write("--- ddsp worker stderr (tail) ---\n")
                self._file.write(stderr_tail)
                if not stderr_tail.endswith("\n"):
                    self._file.write("\n")
                self._file.flush()

    def _maybe_heartbeat(self) -> None:
        interval = float(os.environ.get("SPDMX_SYNTH_LOG_HEARTBEAT_SEC", "600"))
        now = time.monotonic()
        if now - self._last_heartbeat < interval:
            return
        self._last_heartbeat = now
        self.info(
            "heartbeat",
            songs_done=self._songs_done,
            tracks_done=self._tracks_done,
        )

    def close(self) -> None:
        self.info(
            "=== synthesis job log closed ===",
            songs_done=self._songs_done,
            tracks_done=self._tracks_done,
        )
        with self._lock:
            self._file.close()
