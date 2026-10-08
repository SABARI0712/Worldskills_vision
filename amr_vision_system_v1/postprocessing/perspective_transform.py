from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


class PerspectiveTransform:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg or {}
        self.enabled = bool(self.cfg.get("enabled", False))
        self.output_size = self._load_output_size(self.cfg.get("output_size", {}))
        self.points = self._load_points(self.cfg.get("points", {}))
        self.matrix = self.compute_matrix() if self.enabled else None

    @staticmethod
    def _load_output_size(cfg: Dict[str, Any]) -> Tuple[int, int]:
        width = int(cfg.get("width", 800))
        height = int(cfg.get("height", 800))
        return max(1, width), max(1, height)

    @staticmethod
    def _load_point(value: Any) -> Optional[Tuple[float, float]]:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            return None
        try:
            return float(value[0]), float(value[1])
        except (TypeError, ValueError):
            return None

    def _load_points(self, points_cfg: Dict[str, Any]) -> Optional[List[Tuple[float, float]]]:
        order = ["top_left", "top_right", "bottom_right", "bottom_left"]
        points: List[Tuple[float, float]] = []
        for key in order:
            point = self._load_point(points_cfg.get(key))
            if point is None:
                return None
            points.append(point)
        return points

    def set_points(self, points: List[Tuple[float, float]]) -> None:
        if len(points) != 4:
            raise ValueError("PerspectiveTransform requires exactly 4 points")
        self.points = points
        self.matrix = self.compute_matrix()

    def compute_matrix(self) -> Optional[np.ndarray]:
        if not self.enabled or self.points is None:
            return None
        src = np.array(self.points, dtype=np.float32)
        dst = np.array(
            [
                [0.0, 0.0],
                [float(self.output_size[0] - 1), 0.0],
                [float(self.output_size[0] - 1), float(self.output_size[1] - 1)],
                [0.0, float(self.output_size[1] - 1)],
            ],
            dtype=np.float32,
        )
        return cv2.getPerspectiveTransform(src, dst)

    def apply(self, frame_bgr: np.ndarray) -> np.ndarray:
        if not self.enabled or self.matrix is None:
            return frame_bgr
        return cv2.warpPerspective(
            frame_bgr,
            self.matrix,
            self.output_size,
            flags=cv2.INTER_LINEAR,
        )
