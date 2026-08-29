"""Tests for synthesis song-pool helpers."""

from __future__ import annotations

import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait


def _double(x: int) -> int:
    time.sleep(0.05)
    return x * 2


def test_apply_result_ready_poll_pattern():
    """multiprocessing ApplyResult exposes ready()/get(), not Future._condition."""
    import multiprocessing as mp

    with mp.Pool(1) as pool:
        async_result = pool.apply_async(_double, (3,))
        assert hasattr(async_result, "ready")
        assert not hasattr(async_result, "_condition")
        deadline = time.time() + 5
        while not async_result.ready():
            assert time.time() < deadline
            time.sleep(0.01)
        assert async_result.get() == 6


def test_thread_future_still_works_with_wait():
    with ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(lambda: 7)
        done, pending = wait({fut}, return_when=FIRST_COMPLETED)
        assert fut in done
        assert fut.result() == 7
