from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Dict, List, Optional
import time

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
        self._ocr_interval = int(interval_val) if interval_val is not None else 20
        self._frame_index = 0

        self._ycfg = ycfg
        self._ccfg = ccfg

        # Threaded OCR — never blocks the main loop
        self._ocr_executor: Optional[ThreadPoolExecutor] = None
        self._ocr_future: Optional[Future] = None
        self._ocr_cache: List[Detection] = []

        # Detection cache to support skipped frames (Track-first propagation)
        self._detector_cache: Dict[str, List[Detection]] = {
            "yolo": [],
            "qr": [],
            "aruco": [],
            "ocr": [],
            "color": [],
            "contour": [],
        }
        self._frames_since_qr = 0
        self._last_detect_time = 0.0

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

        t_start = time.perf_counter()
        self._frame_index += 1

        # =========================================================
        # 1. OPTIMIZATION: DETECTOR SCHEDULER & PERFORMANCE MANAGER
        # =========================================================
        # Basic scheduled frequencies (modulo frame intervals):
        # ArUco: every frame
        # QR: every 2 frames
        # YOLO: every 3 frames
        # Contour: every 5 frames
        # Color: every 5 frames
        # OCR: every 20 frames (or dynamic based on config)
        
        # Adaptive performance load balancing (Step 10):
        if self._last_detect_time > 0.05:
            # Under load, dynamically back off frequency of heavy detectors to preserve FPS
            yolo_mod = 6
            qr_mod = 4
            ocr_mod = 40
            contour_mod = 10
            color_mod = 10
        else:
            yolo_mod = 3
            qr_mod = 2
            ocr_mod = self._ocr_interval
            contour_mod = 5
            color_mod = 5

        run_aruco = True
        run_qr = (self._frame_index % qr_mod == 0)
        run_yolo = (self._frame_index % yolo_mod == 0)
        run_ocr = (self._frame_index % ocr_mod == 0)
        run_contour = (self._frame_index % contour_mod == 0)
        run_color = (self._frame_index % color_mod == 0)

        dets: List[Detection] = []

        # =========================================================
        # 2. DETECTOR EXECUTION ORDER (PRIORITY MAPPING, Step 12)
        # =========================================================

        # ─── Priority 1: ArUco (runs every frame) ───
        aruco_dets = []
        if self._aruco:
            aruco_dets = self._aruco.detect(frame_bgr)
            self._detector_cache["aruco"] = aruco_dets
        dets.extend(aruco_dets)

        # ─── Priority 2: YOLO (runs scheduled) ───
        # YOLO runs first because it provides bounding boxes for ROI-based QR/OCR
        yolo_dets = []
        if self._yolo and self.yolo_enabled:
            if run_yolo:
                yolo_dets = self._yolo.detect(frame_bgr, conf=self.confidence)
                self._detector_cache["yolo"] = yolo_dets
            else:
                # Carry over previous YOLO detections to maintain smooth CentroidTracker updates
                yolo_dets = self._detector_cache.get("yolo", [])
        dets.extend(yolo_dets)

        # Extract YOLO/Aruco regions of interest (ROI extraction, Step 2)
        candidate_rois = []
        for yd in yolo_dets:
            candidate_rois.append(yd.bbox_xyxy)
        for ad in aruco_dets:
            candidate_rois.append(ad.bbox_xyxy)

        # ─── Priority 3: QR (runs scheduled & ROI-based) ───
        qr_dets = []
        if self._qr:
            if not self._detector_cache.get("qr"):
                self._frames_since_qr += 1
            else:
                self._frames_since_qr = 0

            # Step 8: Only upscale if QR has failed to detect for 10 frames
            upscale = (self._frames_since_qr > 10)

            if run_qr:
                # ROI-based QR execution inside YOLO bounding boxes
                if candidate_rois:
                    for roi_box in candidate_rois:
                        x1, y1, x2, y2 = map(int, roi_box)
                        h, w = frame_bgr.shape[:2]
                        # Pad cropped region slightly for complete edge reading
                        px1, py1 = max(0, x1 - 10), max(0, y1 - 10)
                        px2, py2 = min(w, x2 + 10), min(h, y2 + 10)

                        if px2 > px1 and py2 > py1:
                            roi_crop = frame_bgr[py1:py2, px1:px2]
                            roi_dets = self._qr.detect(roi_crop, upscale=upscale)
                            # Map coordinates back to the global frame
                            for rd in roi_dets:
                                rx1, ry1, rx2, ry2 = rd.bbox_xyxy
                                rd.bbox_xyxy = (rx1 + px1, ry1 + py1, rx2 + px1, ry2 + py1)
                                qr_dets.append(rd)

                # Fallback to full frame scans occasionally or if no ROIs exist
                if not candidate_rois or (self._frame_index % 4 == 0):
                    full_qr_dets = self._qr.detect(frame_bgr, upscale=upscale)
                    for fd in full_qr_dets:
                        if not any(np.array_equal(fd.bbox_xyxy, q.bbox_xyxy) for q in qr_dets):
                            qr_dets.append(fd)
                
                self._detector_cache["qr"] = qr_dets
            else:
                # Carry over cached QR detections
                qr_dets = self._detector_cache.get("qr", [])
        dets.extend(qr_dets)

        # ─── Priority 4: OCR (runs asynchronously & ROI-based) ───
        ocr_dets = []
        if self._ocr is not None and self._ocr_executor is not None:
            # Harvest completed background results
            if self._ocr_future is not None and self._ocr_future.done():
                try:
                    self._ocr_cache = self._ocr_future.result()
                except Exception:
                    self._ocr_cache = []
                self._ocr_future = None
                self._detector_cache["ocr"] = self._ocr_cache

            # Submit task conditionally
            if self._ocr_future is None and run_ocr:
                ocr_crops = []
                if candidate_rois:
                    for roi_box in candidate_rois:
                        x1, y1, x2, y2 = map(int, roi_box)
                        h, w = frame_bgr.shape[:2]
                        px1, py1 = max(0, x1 - 5), max(0, y1 - 5)
                        px2, py2 = min(w, x2 + 5), min(h, y2 + 5)
                        if px2 > px1 and py2 > py1:
                            crop = frame_bgr[py1:py2, px1:px2].copy()
                            ocr_crops.append((crop, (px1, py1)))

                if not ocr_crops:
                    # Run on full frame only if no ROIs exist
                    ocr_crops.append((frame_bgr.copy(), (0, 0)))

                def run_async_ocr(crops):
                    all_dets = []
                    for crop_img, (ox, oy) in crops:
                        crop_dets = self._ocr.detect(crop_img)
                        for cd in crop_dets:
                            rx1, ry1, rx2, ry2 = cd.bbox_xyxy
                            cd.bbox_xyxy = (rx1 + ox, ry1 + oy, rx2 + ox, ry2 + oy)
                            all_dets.append(cd)
                    return all_dets

                self._ocr_future = self._ocr_executor.submit(run_async_ocr, ocr_crops)

            ocr_dets = self._detector_cache.get("ocr", [])
        dets.extend(ocr_dets)

        # ─── Priority 5: Low-level Contour / Color detectors ───
        # Build protected regions to avoid low-level noise matching
        protected_regions = []
        for d in dets:
            if d.source in ["aruco", "qr", "yolo", "ocr"]:
                protected_regions.append(d.bbox_xyxy)

        # Color
        color_dets = []
        if self._color:
            if run_color:
                color_dets = self._color.detect(frame_bgr, protected_regions=protected_regions)
                self._detector_cache["color"] = color_dets
            else:
                color_dets = self._detector_cache.get("color", [])
        dets.extend(color_dets)

        # Contour
        contour_dets = []
        if self._contour:
            if run_contour:
                contour_dets = self._contour.detect(frame_bgr)
                self._detector_cache["contour"] = contour_dets
            else:
                contour_dets = self._detector_cache.get("contour", [])
        dets.extend(contour_dets)

        # Filter very low confidence detections
        dets = [d for d in dets if d.confidence >= self.confidence]

        # Update last execution time for performance load balancer
        self._last_detect_time = time.perf_counter() - t_start

        return dets

    # ------------------------------------------------------------------
    def close(self) -> None:
        """Shut down the background OCR thread cleanly on pipeline exit."""
        if self._ocr_executor is not None:
            self._ocr_executor.shutdown(wait=False, cancel_futures=True)
            self._ocr_executor = None
