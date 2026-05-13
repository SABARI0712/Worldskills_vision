from __future__ import annotations

import time
from typing import Optional, Tuple

import cv2

from .utils import try_set_cap_prop


class USBCamera:
    def __init__(
        self,
        index: int = 0,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[float] = None,
        reconnect: bool = True,
        reconnect_wait_s: float = 0.5,
    ) -> None:
        self.index = index
        self.width = width
        self.height = height
        self.fps = fps
        self.reconnect = reconnect
        self.reconnect_wait_s = reconnect_wait_s
        self.cap: Optional[cv2.VideoCapture] = None
        self._open()

    def _open(self) -> None:
        self.cap = cv2.VideoCapture(self.index)
        if self.cap is None:
            return
        try_set_cap_prop(self.cap, cv2.CAP_PROP_FRAME_WIDTH, self.width)
        try_set_cap_prop(self.cap, cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        try_set_cap_prop(self.cap, cv2.CAP_PROP_FPS, self.fps)
        try:
            if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
                try_set_cap_prop(self.cap, cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

    def read(self) -> Tuple[bool, Optional["cv2.Mat"]]:
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

        if self.reconnect:
            self.close()
        return False, None

    def close(self) -> None:
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = None

