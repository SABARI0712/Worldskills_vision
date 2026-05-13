from __future__ import annotations

import threading
import time
from typing import Any, Optional, Tuple


class ThreadedCamera:
    def __init__(self, camera: Any, poll_interval_s: float = 0.002) -> None:
        self.camera = camera
        self.poll_interval_s = poll_interval_s
        self._lock = threading.Lock()
        self._latest_ok = False
        self._latest_frame: Optional[Any] = None
        self._stop = False
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def _poll(self) -> None:
        while not self._stop:
            ok, frame = self.camera.read()
            with self._lock:
                self._latest_ok = ok
                self._latest_frame = frame.copy() if (ok and frame is not None) else frame
            time.sleep(self.poll_interval_s)

    def read(self) -> Tuple[bool, Optional[Any]]:
        with self._lock:
            if self._latest_frame is None:
                return self._latest_ok, None
            return self._latest_ok, self._latest_frame.copy()

    def close(self) -> None:
        self._stop = True
        try:
            self._thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            self.camera.close()
        except Exception:
            pass
