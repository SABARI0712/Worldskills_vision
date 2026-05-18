from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Dict, List, Optional

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
        ocr_cfg = ccfg.get("ocr") or {}
        interval_val = ocr_cfg.get("interval")
        self._ocr_interval = int(interval_val) if interval_val is not None else 5
        self._frame_index = 0

        self._ycfg = ycfg
        self._ccfg = ccfg

        # Threaded OCR — never blocks the main loop
        self._ocr_executor: Optional[ThreadPoolExecutor] = None
        self._ocr_future: Optional[Future] = None
        self._ocr_cache: List[Detection] = []

    # ------------------------------------------------------------------
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
                # Single dedicated background thread — OCR result is always cached,
                # the main pipeline never waits for it.
                self._ocr_executor = ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="ocr_worker"
                )

            if bool(((self._ccfg.get("color") or {}).get("enabled", False))):
                from .classical.color_detector import ColorDetector
                self._color = ColorDetector(self._ccfg.get("color") or {})

            if bool(((self._ccfg.get("contour") or {}).get("enabled", False))):
                from .classical.contour_detector import ContourDetector
                self._contour = ContourDetector(self._ccfg.get("contour") or {})

    # ------------------------------------------------------------------
    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        if not any([self._yolo, self._aruco, self._qr, self._ocr, self._color, self._contour]):
            self.warmup()

        dets: List[Detection] = []

        # === High reliability semantic detectors first ===
        if self._aruco:
            dets.extend(self._aruco.detect(frame_bgr))
        if self._qr:
            dets.extend(self._qr.detect(frame_bgr))
        if self._yolo and self.yolo_enabled:
            dets.extend(self._yolo.detect(frame_bgr, conf=self.confidence))

        # === OCR: fully non-blocking threaded execution ===
        # The main loop never waits for OCR. Results arrive in the next few
        # frames and are cached until the next successful run.
        if self._ocr is not None and self._ocr_executor is not None:
            # Harvest completed result without blocking
            if self._ocr_future is not None and self._ocr_future.done():
                try:
                    self._ocr_cache = self._ocr_future.result()
                except Exception:
                    self._ocr_cache = []
                self._ocr_future = None

            # Submit new task every N frames, only when thread is idle
            if self._ocr_future is None and (self._frame_index % self._ocr_interval) == 0:
                # Copy frame so main loop and OCR thread don't share memory
                self._ocr_future = self._ocr_executor.submit(
                    self._ocr.detect, frame_bgr.copy()
                )

            # Always use last known OCR results — zero latency cost
            dets.extend(self._ocr_cache)

        self._frame_index += 1

        # === Build protected regions for lower-level detectors ===
        protected_regions = []
        for d in dets:
            if d.source in ["aruco", "qr", "yolo", "ocr"]:
                protected_regions.append(d.bbox_xyxy)

        # === Low priority detectors (semantically protected) ===
        if self._color:
            dets.extend(
                self._color.detect(
                    frame_bgr,
                    protected_regions=protected_regions,
                )
            )
        if self._contour:
            dets.extend(self._contour.detect(frame_bgr))

        # Remove very low confidence detections
        dets = [d for d in dets if d.confidence >= self.confidence]

        return dets

    # ------------------------------------------------------------------
    def close(self) -> None:
        """Shut down the background OCR thread cleanly on pipeline exit."""
        if self._ocr_executor is not None:
            self._ocr_executor.shutdown(wait=False, cancel_futures=True)
            self._ocr_executor = None
