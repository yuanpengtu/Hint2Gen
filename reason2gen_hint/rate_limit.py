from __future__ import annotations

import threading
import time
from collections import deque


class RateLimiter:
    def __init__(self, rpm: int = 600):
        self.rpm = max(1, int(rpm))
        self.window = 60.0
        self.ts = deque()
        self.tsr = self.ts
        self._lock = threading.Lock()

    def __getattr__(self, name: str):
        if name == "tsr":
            self.tsr = self.ts
            return self.tsr
        raise AttributeError(name)

    def acquire(self) -> None:
        while True:
            now = time.monotonic()
            with self._lock:
                while self.ts and now - self.ts[0] >= self.window:
                    self.ts.popleft()
                if len(self.ts) < self.rpm:
                    self.ts.append(now)
                    self.tsr = self.ts
                    return
                wait = self.window - (now - self.ts[0])
            time.sleep(max(0.001, min(wait, 0.25)))
