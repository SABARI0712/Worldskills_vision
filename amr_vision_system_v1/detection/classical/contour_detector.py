from __future__ import annotations

from typing import Any, Dict, List

import cv2
import numpy as np

from ..types import Detection


class ContourDetector:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.min_area = float(cfg.get("min_area", 2000))

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 60, 180)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dets: List[Detection] = []
        h, w = frame_bgr.shape[:2]

        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < self.min_area:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            x1, y1, x2, y2 = x, y, x + bw, y + bh
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))
            dets.append(
                Detection(
                    label="contour",
                    confidence=1.0,
                    bbox_xyxy=(x1, y1, x2, y2),
                    source="contour",
                    meta={"area": area},
                )
            )
        return dets

