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
        self._ocr = None

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
            if bool(((self._ccfg.get("ocr") or {}).get("enabled", False))):
                from .classical.ocr_detector import OCRDetector

                self._ocr = OCRDetector(self._ccfg.get("ocr") or {})
            if bool(((self._ccfg.get("color") or {}).get("enabled", False))):
                from .classical.color_detector import ColorDetector

                self._color = ColorDetector(self._ccfg.get("color") or {})
            if bool(((self._ccfg.get("contour") or {}).get("enabled", False))):
                from .classical.contour_detector import ContourDetector

                self._contour = ContourDetector(self._ccfg.get("contour") or {})

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        if not any([self._yolo, self._aruco, self._qr, self._ocr, self._color, self._contour]):
            self.warmup()

        dets: List[Detection] = []

        # High reliability first
        if self._aruco:
            dets.extend(self._aruco.detect(frame_bgr))
        if self._qr:
            dets.extend(self._qr.detect(frame_bgr))

        # Main object detection
        if self._yolo and self.yolo_enabled:
            dets.extend(self._yolo.detect(frame_bgr, conf=self.confidence))

        # OCR (already filtered)
        if self._ocr:
            dets.extend(self._ocr.detect(frame_bgr))

        # Low priority
        if self._color:
            dets.extend(self._color.detect(frame_bgr))
        if self._contour:
            dets.extend(self._contour.detect(frame_bgr))

        # Remove very low confidence detections
        dets = [d for d in dets if d.confidence >= 0.5]

        return dets

