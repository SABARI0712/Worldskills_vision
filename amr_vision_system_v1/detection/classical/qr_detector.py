from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

from ..types import Detection


class QRDetector:
    def __init__(self) -> None:
        self._detector = cv2.QRCodeDetector()

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        data, points, _ = self._detector.detectAndDecode(frame_bgr)
        if not data or points is None:
            return []

        pts = points[0]
        xs = pts[:, 0]
        ys = pts[:, 1]
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

        return [
            Detection(
                label=f"qr:{data}",
                confidence=1.0,
                bbox_xyxy=(x1, y1, x2, y2),
                source="qr",
                meta={"data": data},
            )
        ]

