from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

from ..types import Detection


class QRDetector:
    def __init__(self) -> None:
        self._detector = cv2.QRCodeDetector()

    def _normalize_points(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=float)
        if pts.ndim == 3 and pts.shape[0] == 1:
            pts = pts[0]
        if pts.ndim == 3 and pts.shape[1] == 1:
            pts = pts.reshape(-1, 2)
        return pts

    def _make_detection(self, pts: np.ndarray, data: str) -> Detection:
        pts = self._normalize_points(pts)
        if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 4:
            raise ValueError("Invalid QR points")

        xs = pts[:, 0]
        ys = pts[:, 1]
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        label = f"qr:{data}" if data else "qr_code"
        confidence = 1.0 if data else 0.6
        return Detection(
            label=label,
            confidence=confidence,
            bbox_xyxy=(x1, y1, x2, y2),
            source="qr",
            meta={"data": data or ""},
        )

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        detections: List[Detection] = []
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        try:
            if hasattr(self._detector, "detectAndDecodeMulti"):
                multi_result = self._detector.detectAndDecodeMulti(gray)
                if not isinstance(multi_result, tuple):
                    return []

                decoded_info = []
                points = None

                if len(multi_result) == 3:
                    first, second, third = multi_result
                    if isinstance(first, bool):
                        decoded_info = second if isinstance(second, (list, tuple)) else []
                        points = third
                    else:
                        decoded_info = first if isinstance(first, (list, tuple)) else []
                        points = second
                elif len(multi_result) == 4:
                    first, second, third, _ = multi_result
                    if isinstance(first, bool):
                        decoded_info = second if isinstance(second, (list, tuple)) else []
                        points = third
                    else:
                        decoded_info = first if isinstance(first, (list, tuple)) else []
                        points = second
                else:
                    return []

                if points is None:
                    return []

                for index, pts in enumerate(points):
                    if pts is None:
                        continue
                    try:
                        detections.append(self._make_detection(pts, decoded_info[index] if index < len(decoded_info) and isinstance(decoded_info, (list, tuple)) else ""))
                    except ValueError:
                        continue
                return detections

            data, points, _ = self._detector.detectAndDecode(gray)
        except cv2.error:
            return []

        if points is None or len(points) == 0:
            return []

        try:
            detections.append(self._make_detection(points[0], data))
        except ValueError:
            return []

        return detections

