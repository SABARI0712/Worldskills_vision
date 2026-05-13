from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np

from ..types import Detection


class ContourDetector:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.min_area = float(cfg.get("min_area", 2000))

    @staticmethod
    def _bbox_area(box: Sequence[float]) -> float:
        x1, y1, x2, y2 = box
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    @staticmethod
    def _intersection(box_a: Sequence[float], box_b: Sequence[float]) -> float:
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        return width * height

    def _iou(self, box_a: Sequence[float], box_b: Sequence[float]) -> float:
        inter = self._intersection(box_a, box_b)
        union = self._bbox_area(box_a) + self._bbox_area(box_b) - inter
        if union <= 0.0:
            return 0.0
        return inter / union

    def _overlaps(self, box: Sequence[float], protected_region: Sequence[float], threshold: float = 0.3) -> bool:
        return self._iou(box, protected_region) >= float(threshold)

    def detect(
        self,
        frame_bgr: np.ndarray,
        protected_regions: Optional[List[Sequence[float]]] = None,
    ) -> List[Detection]:
        protected_regions = protected_regions or []
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
            bbox = [x1, y1, x2, y2]
            if any(self._overlaps(bbox, region) for region in protected_regions):
                continue
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

