from __future__ import annotations

from typing import Any, Dict, List

import cv2
import numpy as np

from ..types import Detection


class ArucoDetector:
    def __init__(self, cfg: Dict[str, Any] | None = None) -> None:
        cfg = cfg or {}
        self.dictionary_name = str(cfg.get("dictionary", "DICT_4X4_50"))
        self.min_marker_area = int(cfg.get("min_marker_area", 100))
        self.corner_refinement = bool(cfg.get("corner_refinement", True))

        self._dictionary = self._make_dictionary(self.dictionary_name)
        self._parameters = self._make_detector_parameters(self.corner_refinement)
        self._detector = self._make_detector(self._dictionary, self._parameters)

    def _make_dictionary(self, dictionary_name: str) -> Any:
        if not hasattr(cv2.aruco, dictionary_name):
            available = [name for name in dir(cv2.aruco) if name.startswith("DICT_")]
            raise ValueError(
                f"Unknown ArUco dictionary '{dictionary_name}'. "
                f"Available dictionaries: {available}"
            )
        dictionary_id = getattr(cv2.aruco, dictionary_name)
        return cv2.aruco.getPredefinedDictionary(dictionary_id)

    def _make_detector_parameters(self, corner_refinement: bool) -> Any:
        if hasattr(cv2.aruco, "DetectorParameters_create"):
            params = cv2.aruco.DetectorParameters_create()
        else:
            params = cv2.aruco.DetectorParameters()

        if corner_refinement:
            if hasattr(params, "cornerRefinementMethod"):
                params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            elif hasattr(params, "cornerRefinement"):
                params.cornerRefinement = True
        return params

    def _make_detector(self, dictionary: Any, parameters: Any) -> Any:
        if hasattr(cv2.aruco, "ArucoDetector"):
            return cv2.aruco.ArucoDetector(dictionary, parameters)
        return None

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        if frame_bgr is None or frame_bgr.size == 0:
            return []

        if self._detector is not None:
            corners, ids, _ = self._detector.detectMarkers(frame_bgr)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(frame_bgr, self._dictionary, parameters=self._parameters)

        if ids is None or len(ids) == 0:
            return []

        detections: List[Detection] = []
        for marker_corners, marker_id in zip(corners, ids):
            pts = marker_corners[0].astype(float)
            x_coords = pts[:, 0]
            y_coords = pts[:, 1]
            x1, y1 = int(x_coords.min()), int(y_coords.min())
            x2, y2 = int(x_coords.max()), int(y_coords.max())
            center_x = int(x_coords.mean())
            center_y = int(y_coords.mean())

            if self.min_marker_area > 0:
                area = float(cv2.contourArea(pts.astype(np.int32)))
                if area < float(self.min_marker_area):
                    continue

            side_lengths = [
                float(np.linalg.norm(pts[(i + 1) % 4] - pts[i]))
                for i in range(4)
            ]
            side_px = float(np.mean(side_lengths))

            detections.append(
                Detection(
                    label=f"aruco:{int(marker_id[0])}",
                    confidence=1.0,
                    bbox_xyxy=(x1, y1, x2, y2),
                    source="aruco",
                    meta={
                        "marker_id": int(marker_id[0]),
                        "center": [center_x, center_y],
                        "side_px": side_px,
                        "corner_refinement": self.corner_refinement,
                    },
                )
            )

        return detections
