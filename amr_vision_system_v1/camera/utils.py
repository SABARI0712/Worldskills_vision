from __future__ import annotations

from typing import Optional

import cv2


def try_set_cap_prop(cap: cv2.VideoCapture, prop: int, value: Optional[float]) -> None:
    if value is None:
        return
    try:
        cap.set(prop, float(value))
    except Exception:
        pass

