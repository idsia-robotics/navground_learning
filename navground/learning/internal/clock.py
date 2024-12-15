from __future__ import annotations

import time


class SyncClock:

    def __init__(self, time_step: float, factor: float = 1.0):
        self._last_time: float | None = None
        self._period = time_step / factor

    def tick(self) -> None:
        now = time.monotonic()
        if self._last_time is not None:
            sleep_time = max(self._last_time + self._period - now, 0.0)
            if sleep_time > 0:
                time.sleep(sleep_time)
                now = time.monotonic()
        self._last_time = now
