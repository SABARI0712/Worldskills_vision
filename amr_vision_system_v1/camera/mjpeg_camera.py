from __future__ import annotations

import time
from typing import Optional, Tuple

import cv2


class MJPEGCamera:
    def __init__(
        self,
        url: str,
        reconnect: bool = True,
        reconnect_wait_s: float = 1.0,
    ) -> None:
        self.url = url
        self.reconnect = reconnect
        self.reconnect_wait_s = reconnect_wait_s
        self.cap: Optional[cv2.VideoCapture] = None
        self._open()

    def _open(self) -> None:
        self.close()
        self.cap = cv2.VideoCapture(self.url)

        if self.cap is None:
            return

        try:
            if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

    def read(self) -> Tuple[bool, Optional[cv2.typing.MatLike]]:
        if self.cap is None or not self.cap.isOpened():
            if not self.reconnect:
                return False, None

            self.close()
            time.sleep(self.reconnect_wait_s)
            self._open()

            if self.cap is None or not self.cap.isOpened():
                return False, None

        ok, frame = self.cap.read()
        if ok:
            return True, frame

        self.close()
        return False, None

    def close(self) -> None:
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass

        self.cap = None
