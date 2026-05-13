from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np

from ..types import Detection


class ColorDetector:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.min_area = float(cfg.get("min_area", 7000))
        self.min_aspect_ratio = float(cfg.get("min_aspect_ratio", 0.2))
        self.max_aspect_ratio = float(cfg.get("max_aspect_ratio", 5.0))
        self.hsv_ranges = cfg.get("hsv_ranges", {}) or {}

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
        if not self.hsv_ranges:
            return []
        protected_regions = protected_regions or []
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        dets: List[Detection] = []
        h, w = frame_bgr.shape[:2]

        for name, bounds in self.hsv_ranges.items():
            try:
                lower, upper = bounds
                lower = np.array(lower, dtype=np.uint8)
                upper = np.array(upper, dtype=np.uint8)
            except Exception:
                continue

            mask = cv2.inRange(hsv, lower, upper)
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = float(cv2.contourArea(cnt))
                if area < self.min_area:
                    continue
                x, y, bw, bh = cv2.boundingRect(cnt)
                if bw <= 0 or bh <= 0:
                    continue
                aspect_ratio = float(bw) / float(bh)
                if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
                    continue
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
                        label=f"color:{name}",
                        confidence=1.0,
                        bbox_xyxy=(x1, y1, x2, y2),
                        source="color",
                        meta={"mask_area": area},
                    )
                )

        return dets

