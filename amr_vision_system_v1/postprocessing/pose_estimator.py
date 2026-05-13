from __future__ import annotations

import cv2
import numpy as np
from typing import Any, Dict, List, Tuple


class PoseEstimator:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.enabled = bool(cfg.get("enabled", True))
        self.min_area = int(cfg.get("min_area", 1000))
        self.use_canny = bool(cfg.get("use_canny", True))
        self.canny_threshold1 = int(cfg.get("canny_threshold1", 50))
        self.canny_threshold2 = int(cfg.get("canny_threshold2", 150))
        self.threshold_value = int(cfg.get("threshold_value", 127))

    def estimate_pose(self, detections: List[Dict[str, Any]], frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Estimate 2D orientation (angle) for tracked semantic objects using cv2.minAreaRect().

        Args:
            detections: List of detection dictionaries with bbox_xyxy, label, source, id, etc.
            frame: Warped/perspective-corrected BGR frame

        Returns:
            Updated detections with 'angle' key added for eligible objects
        """
        if not self.enabled:
            return detections

        updated_detections = []

        for det in detections:
            # Only estimate pose for semantic objects that are tracked
            if not self._is_eligible_for_pose(det):
                updated_detections.append(det)
                continue

            angle = self._compute_angle(det, frame)
            if angle is not None:
                det["angle"] = angle
            updated_detections.append(det)

        return updated_detections

    def _is_eligible_for_pose(self, det: Dict[str, Any]) -> bool:
        """Check if detection is eligible for pose estimation."""
        # Must be tracked (have id)
        if "id" not in det:
            return False

        # Must be semantic object (not QR/ArUco)
        source = det.get("source", "")
        if source in ["qr", "aruco"]:
            return False

        # Must have valid bbox
        bbox = det.get("bbox_xyxy")
        if not bbox or len(bbox) != 4:
            return False

        return True

    def _compute_angle(self, det: Dict[str, Any], frame: np.ndarray) -> float | None:
        """Compute orientation angle using minAreaRect."""
        x1, y1, x2, y2 = det["bbox_xyxy"]

        # Extract ROI
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Edge detection or thresholding
        if self.use_canny:
            edges = cv2.Canny(gray, self.canny_threshold1, self.canny_threshold2)
        else:
            _, edges = cv2.threshold(gray, self.threshold_value, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Find largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)

        if area < self.min_area:
            return None

        # Get min area rect and choose the longest edge for stable heading
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)

        angle = self._angle_from_box(box)
        return angle

    def _angle_from_box(self, box: np.ndarray) -> float:
        """Compute a stable 0-180 orientation from a rotated rectangle."""
        pts = np.array(box, dtype=np.float32)
        edges = np.roll(pts, -1, axis=0) - pts
        lengths = np.linalg.norm(edges, axis=1)
        if lengths.size == 0:
            return 0.0

        longest_edge = edges[int(np.argmax(lengths))]
        angle = np.degrees(np.arctan2(longest_edge[1], longest_edge[0]))
        
        angle = angle % 180
        if angle < 0:
            angle += 180
        return round(angle, 1)