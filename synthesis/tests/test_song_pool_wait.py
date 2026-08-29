"""Tests for synthesis song-pool helpers."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait


def test_thread_future_still_works_with_wait():
    with ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(lambda: 7)
        done, pending = wait({fut}, return_when=FIRST_COMPLETED)
        assert fut in done
        assert fut.result() == 7
