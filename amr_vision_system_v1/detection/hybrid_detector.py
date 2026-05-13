from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from .types import Detection


class HybridDetector:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.mode = str(cfg.get("mode", "hybrid")).lower()
        self.confidence = float(cfg.get("confidence", 0.5))

        self._yolo = None
        self._qr = None
        self._aruco = None
        self._color = None
        self._contour = None

        ycfg = (cfg.get("yolo") or {}) if cfg else {}
        ccfg = (cfg.get("classical") or {}) if cfg else {}

        self.yolo_enabled = bool(ycfg.get("enabled", True))
        self.classical_enabled = bool(ccfg.get("enabled", True))

        self._ycfg = ycfg
        self._ccfg = ccfg

    def warmup(self) -> None:
        if self.yolo_enabled and self.mode in ("yolo", "hybrid"):
            from .ml.yolo_detector import YOLODetector

            self._yolo = YOLODetector(self._ycfg)
            self._yolo.warmup()

        if self.classical_enabled and self.mode in ("classical", "hybrid"):
            if bool(((self._ccfg.get("qr") or {}).get("enabled", True))):
                from .classical.qr_detector import QRDetector

                self._qr = QRDetector()
            if bool(((self._ccfg.get("aruco") or {}).get("enabled", False))):
                from .classical.aruco_detector import ArucoDetector

                self._aruco = ArucoDetector(self._ccfg.get("aruco") or {})
            if bool(((self._ccfg.get("color") or {}).get("enabled", False))):
                from .classical.color_detector import ColorDetector

                self._color = ColorDetector(self._ccfg.get("color") or {})
            if bool(((self._ccfg.get("contour") or {}).get("enabled", False))):
                from .classical.contour_detector import ContourDetector

                self._contour = ContourDetector(self._ccfg.get("contour") or {})

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        if self._yolo is None and self._qr is None and self._color is None and self._contour is None:
            self.warmup()

        dets: List[Detection] = []
        occupied_regions: List[List[float]] = []

        if self.mode in ("classical", "hybrid") and self.classical_enabled:
            if self._aruco is not None:
                aruco_dets = self._aruco.detect(frame_bgr)
                dets.extend(aruco_dets)
                occupied_regions.extend(
                    [list(det.bbox_xyxy) for det in aruco_dets]
                )

            if self._qr is not None:
                qr_dets = self._qr.detect(frame_bgr)
                dets.extend(qr_dets)
                occupied_regions.extend(
                    [list(det.bbox_xyxy) for det in qr_dets]
                )

        if self.mode in ("yolo", "hybrid") and self.yolo_enabled and self._yolo is not None:
            yolo_dets = self._yolo.detect(frame_bgr, conf=self.confidence)
            dets.extend(yolo_dets)
            occupied_regions.extend(
                [list(det.bbox_xyxy) for det in yolo_dets]
            )

        if self.mode in ("classical", "hybrid") and self.classical_enabled:
            if self._color is not None:
                dets.extend(self._color.detect(frame_bgr, occupied_regions))
            if self._contour is not None:
                dets.extend(self._contour.detect(frame_bgr, occupied_regions))

        return dets

