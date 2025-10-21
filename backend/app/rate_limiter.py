from __future__ import annotations

import time
from collections import deque
from threading import Lock
from typing import Deque, Dict


class RateLimiter:
    """Simple in-memory sliding window rate limiter."""

    def __init__(self, limit: int, window_seconds: int) -> None:
        self.limit = max(1, limit)
        self.window = max(1, window_seconds)
        self._hits: Dict[str, Deque[float]] = {}
        self._lock = Lock()

    def allow(self, key: str) -> bool:
        """Register a hit for the given key. Returns False when over limit."""
        now = time.time()
        with self._lock:
            bucket = self._hits.setdefault(key, deque())
            while bucket and now - bucket[0] > self.window:
                bucket.popleft()
            if len(bucket) >= self.limit:
                return False
            bucket.append(now)
            return True
