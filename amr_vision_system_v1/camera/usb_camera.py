from __future__ import annotations

import time
from typing import Optional, Tuple

import cv2

from .utils import try_set_cap_prop


class USBCamera:
    def __init__(
        self,
        index: int = 0,
        width: Optional[int] = 1280,
        height: Optional[int] = 720,
        fps: Optional[float] = 30,
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

        # V4L2 backend works better on Linux + Orbbec
        self.cap = cv2.VideoCapture(self.index, cv2.CAP_V4L2)

        if self.cap is None:
            return

        # Resolution
        try_set_cap_prop(self.cap, cv2.CAP_PROP_FRAME_WIDTH, self.width)
        try_set_cap_prop(self.cap, cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        try_set_cap_prop(self.cap, cv2.CAP_PROP_FPS, self.fps)

        # LOW LATENCY
        try:
            if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
                try_set_cap_prop(self.cap, cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        # =========================
        # GEMINI E OPTIMIZATION
        # =========================

        try:
            # Disable autofocus
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

            # Manual focus
            self.cap.set(cv2.CAP_PROP_FOCUS, 10)

            # Disable aggressive auto exposure
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)

            # Exposure tuning
            self.cap.set(cv2.CAP_PROP_EXPOSURE, -6)

            # Lower gain = less noise
            self.cap.set(cv2.CAP_PROP_GAIN, 0)

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

        self.close()
        return False, None

    def close(self) -> None:

        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass

        self.cap = None