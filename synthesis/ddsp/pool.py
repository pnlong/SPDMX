"""Persistent multi-GPU pool of TF DDSP serve workers (JSONL over stdin/stdout)."""

from __future__ import annotations

import atexit
import collections
import json
import os
import queue
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from synthesis.ddsp.env import (
    ddsp_python_executable,
    ddsp_worker_env,
    parse_ddsp_gpu_ids,
)

_READY_TIMEOUT_SEC = float(os.environ.get("SPDMX_DDSP_POOL_READY_TIMEOUT", "600"))
_GLOBAL_POOL: "DdspWorkerPool | None" = None
_GLOBAL_LOCK = threading.Lock()
_ATEXIT_REGISTERED = False


def ddsp_oneshot_enabled() -> bool:
    return os.environ.get("SPDMX_DDSP_ONESHOT") == "1"


@dataclass
class _ServeWorker:
    gpu_id: str
    proc: subprocess.Popen
    lock: threading.Lock = field(default_factory=threading.Lock)
    next_id: int = 0
    _stdout_q: queue.Queue = field(default_factory=queue.Queue)
    _stderr_lines: collections.deque = field(
        default_factory=lambda: collections.deque(maxlen=400)
    )
    _io_started: bool = False

    def start_io(self) -> None:
        """Drain stdout/stderr on background threads.

        Critical: leaving ``stderr=PIPE`` unread deadlocks TF during model load
        once the OS pipe buffer fills.
        """
        if self._io_started:
            return
        self._io_started = True
        threading.Thread(
            target=self._drain_stdout, name=f"ddsp-stdout-{self.gpu_id}", daemon=True
        ).start()
        threading.Thread(
            target=self._drain_stderr, name=f"ddsp-stderr-{self.gpu_id}", daemon=True
        ).start()

    def _drain_stdout(self) -> None:
        try:
            assert self.proc.stdout is not None
            while True:
                line = self.proc.stdout.readline()
                if line:
                    self._stdout_q.put(line)
                    continue
                if self.proc.poll() is not None:
                    break
                time.sleep(0.01)
        except Exception as exc:
            self._stdout_q.put(exc)
        finally:
            self._stdout_q.put(None)

    def _drain_stderr(self) -> None:
        try:
            assert self.proc.stderr is not None
            while True:
                line = self.proc.stderr.readline()
                if line:
                    self._stderr_lines.append(line)
                    # Surface worker status lines (weight load / AR patch) to the
                    # parent terminal — otherwise they stay trapped in the pipe.
                    if line.startswith("[ddsp"):
                        print(line, end="", flush=True)
                    continue
                if self.proc.poll() is not None:
                    break
                time.sleep(0.01)
        except Exception:
            pass

    def stderr_tail(self, limit: int = 2000) -> str:
        return "".join(self._stderr_lines)[-limit:]

    def readline(self, *, timeout_sec: float) -> str:
        deadline = time.monotonic() + timeout_sec
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"DDSP serve worker (GPU {self.gpu_id}) timed out after "
                    f"{timeout_sec:.0f}s\nstderr:\n{self.stderr_tail()}"
                )
            try:
                item = self._stdout_q.get(timeout=min(0.5, remaining))
            except queue.Empty:
                if self.proc.poll() is not None:
                    raise RuntimeError(
                        f"DDSP serve worker (GPU {self.gpu_id}) exited "
                        f"(code {self.proc.returncode})\nstderr:\n{self.stderr_tail()}"
                    )
                continue
            if item is None:
                raise RuntimeError(
                    f"DDSP serve worker (GPU {self.gpu_id}) closed stdout "
                    f"(code {self.proc.returncode})\nstderr:\n{self.stderr_tail()}"
                )
            if isinstance(item, Exception):
                raise RuntimeError(
                    f"DDSP serve worker (GPU {self.gpu_id}) stdout reader failed: {item}\n"
                    f"stderr:\n{self.stderr_tail()}"
                ) from item
            return item

    def request(self, payload: dict[str, Any], *, timeout_sec: float) -> dict:
        with self.lock:
            return self._request_locked(payload, timeout_sec=timeout_sec)

    def _request_locked(self, payload: dict[str, Any], *, timeout_sec: float) -> dict:
        self.next_id += 1
        req_id = self.next_id
        body = dict(payload)
        body["id"] = req_id
        line = json.dumps(body) + "\n"
        try:
            assert self.proc.stdin is not None
            self.proc.stdin.write(line)
            self.proc.stdin.flush()
        except BrokenPipeError as exc:
            raise RuntimeError(
                f"DDSP serve worker (GPU {self.gpu_id}) pipe broken\n"
                f"stderr:\n{self.stderr_tail()}"
            ) from exc

        deadline = time.monotonic() + timeout_sec
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"DDSP serve worker (GPU {self.gpu_id}) timed out after "
                    f"{timeout_sec:.0f}s waiting for id={req_id}\n"
                    f"stderr:\n{self.stderr_tail()}"
                )
            raw = self.readline(timeout_sec=remaining).strip()
            if not raw:
                continue
            try:
                status = json.loads(raw)
            except json.JSONDecodeError:
                # TF/absl may leak non-JSON lines to stdout; ignore.
                continue
            if status.get("id") == req_id:
                return status
            if status.get("ready"):
                continue

    def shutdown(self) -> None:
        if self.proc.poll() is not None:
            return
        try:
            with self.lock:
                if self.proc.poll() is not None:
                    return
                assert self.proc.stdin is not None
                self.proc.stdin.write(
                    json.dumps({"id": 0, "command": "shutdown"}) + "\n"
                )
                self.proc.stdin.flush()
                self.proc.stdin.close()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=10)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass


class DdspWorkerPool:
    """One long-lived ``worker serve`` process per GPU id."""

    def __init__(self, workers: list[_ServeWorker]):
        if not workers:
            raise ValueError("DdspWorkerPool requires at least one worker")
        self._workers = workers
        self._rr = 0
        self._rr_lock = threading.Lock()
        self._closed = False

    @property
    def size(self) -> int:
        return len(self._workers)

    @classmethod
    def start(
        cls,
        *,
        gpu_ids: list[str] | None = None,
        ready_timeout_sec: float | None = None,
        preload: str | None = None,
    ) -> "DdspWorkerPool":
        python = ddsp_python_executable()
        ids = gpu_ids if gpu_ids is not None else parse_ddsp_gpu_ids()
        timeout = float(
            ready_timeout_sec
            if ready_timeout_sec is not None
            else (1800 if preload == "midi_ddsp" else _READY_TIMEOUT_SEC)
        )
        preload = (preload or os.environ.get("SPDMX_DDSP_PRELOAD") or "").strip()
        workers: list[_ServeWorker] = []
        print(
            f"Starting {len(ids)} DDSP serve worker(s)"
            + (f" (preload {preload})" if preload else "")
            + "...",
            flush=True,
        )
        try:
            for gpu_id in ids:
                env = ddsp_worker_env(cuda_visible_devices=gpu_id)
                env["PYTHONUNBUFFERED"] = "1"
                if preload:
                    env["SPDMX_DDSP_PRELOAD"] = preload
                proc = subprocess.Popen(
                    [str(python), "-m", "synthesis.ddsp.worker", "serve"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    env=env,
                )
                worker = _ServeWorker(gpu_id=gpu_id, proc=proc)
                worker.start_io()
                workers.append(worker)
            for worker in workers:
                print(
                    f"Waiting for DDSP GPU {worker.gpu_id}"
                    + (" — MIDI-DDSP weight load can take several minutes" if preload else "")
                    + "...",
                    flush=True,
                )
                deadline = time.monotonic() + timeout
                ready = False
                while time.monotonic() < deadline:
                    if worker.proc.poll() is not None:
                        raise RuntimeError(
                            f"DDSP serve worker failed to start (GPU {worker.gpu_id}): "
                            f"{worker.stderr_tail()}"
                        )
                    remaining = deadline - time.monotonic()
                    try:
                        line = worker.readline(timeout_sec=min(2.0, max(0.05, remaining)))
                    except TimeoutError:
                        continue
                    try:
                        msg = json.loads(line.strip())
                    except json.JSONDecodeError:
                        continue
                    if msg.get("ready"):
                        ready = True
                        break
                if not ready:
                    worker.shutdown()
                    raise TimeoutError(
                        f"DDSP serve worker (GPU {worker.gpu_id}) not ready within "
                        f"{timeout:.0f}s\nstderr:\n{worker.stderr_tail()}"
                    )
                print(f"DDSP GPU {worker.gpu_id} ready.", flush=True)
        except Exception as exc:
            from synthesis.job_log import get_job_log

            log = get_job_log()
            if log is not None:
                log.fatal(exc, where="ddsp pool startup")
            for w in workers:
                w.shutdown()
            raise

        return cls(workers)

    def submit(self, payload: dict[str, Any], *, timeout_sec: float) -> dict:
        if self._closed:
            raise RuntimeError("DdspWorkerPool is closed")
        start = 0
        with self._rr_lock:
            start = self._rr
            self._rr += 1
        n = len(self._workers)
        for offset in range(n):
            worker = self._workers[(start + offset) % n]
            if worker.lock.acquire(blocking=False):
                try:
                    return worker._request_locked(payload, timeout_sec=timeout_sec)
                finally:
                    worker.lock.release()
        worker = self._workers[start % n]
        return worker.request(payload, timeout_sec=timeout_sec)

    def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        for worker in self._workers:
            worker.shutdown()


def get_ddsp_pool(*, preload: str | None = None) -> DdspWorkerPool:
    """Process-global pool (started lazily)."""
    global _GLOBAL_POOL, _ATEXIT_REGISTERED
    with _GLOBAL_LOCK:
        if _GLOBAL_POOL is None or _GLOBAL_POOL._closed:
            _GLOBAL_POOL = DdspWorkerPool.start(preload=preload)
            if not _ATEXIT_REGISTERED:
                atexit.register(shutdown_ddsp_pool)
                _ATEXIT_REGISTERED = True
        return _GLOBAL_POOL


def shutdown_ddsp_pool() -> None:
    global _GLOBAL_POOL
    with _GLOBAL_LOCK:
        if _GLOBAL_POOL is not None:
            _GLOBAL_POOL.shutdown()
            _GLOBAL_POOL = None


def ensure_ddsp_pool(*, preload: str | None = None) -> DdspWorkerPool | None:
    """Start the pool unless oneshot mode is enabled."""
    if ddsp_oneshot_enabled():
        return None
    return get_ddsp_pool(preload=preload)
