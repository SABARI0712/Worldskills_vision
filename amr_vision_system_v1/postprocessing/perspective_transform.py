from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


class PerspectiveTransform:
    """Estimates and applies a perspective warp for board normalization."""

    def __init__(self) -> None:
        self.matrix: Optional[np.ndarray] = None
        self.inverse: Optional[np.ndarray] = None

    def estimate_transform(
        self,
        source_points: List[Tuple[float, float]],
        destination_points: List[Tuple[float, float]],
    ) -> None:
        if len(source_points) != 4 or len(destination_points) != 4:
            raise ValueError("Exactly 4 source and destination points are required")
        self.matrix = cv2.getPerspectiveTransform(
            np.array(source_points, dtype=np.float32),
            np.array(destination_points, dtype=np.float32),
        )
        self.inverse = cv2.getPerspectiveTransform(
            np.array(destination_points, dtype=np.float32),
            np.array(source_points, dtype=np.float32),
        )

    def warp(self, frame: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        if self.matrix is None:
            raise RuntimeError("Perspective transform matrix has not been estimated")
        return cv2.warpPerspective(frame, self.matrix, size)

    def unwarp(self, frame: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        if self.inverse is None:
            raise RuntimeError("Inverse perspective transform matrix has not been estimated")
        return cv2.warpPerspective(frame, self.inverse, size)
