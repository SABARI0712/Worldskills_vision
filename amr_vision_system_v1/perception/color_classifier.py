from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import cv2
import numpy as np


class ColorClassifier:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.hsv_ranges = cfg.get("hsv_ranges", {}) or {}

    def classify(self, frame_bgr: np.ndarray, bbox_xyxy: Sequence[float]) -> Optional[str]:
        x1, y1, x2, y2 = map(int, bbox_xyxy)
        h, w = frame_bgr.shape[:2]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))
        if x2 <= x1 or y2 <= y1:
            return None
        roi = frame_bgr[y1:y2, x1:x2]
        if roi.size == 0:
            return None
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        max_count = 0
        dominant_color = None
        for name, bounds in self.hsv_ranges.items():
            try:
                lower, upper = bounds
                lower = np.array(lower, dtype=np.uint8)
                upper = np.array(upper, dtype=np.uint8)
            except Exception:
                continue
            mask = cv2.inRange(hsv, lower, upper)
            count = cv2.countNonZero(mask)
            if count > max_count:
                max_count = count
                dominant_color = name
        return dominant_color