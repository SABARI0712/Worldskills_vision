from __future__ import annotations

from typing import List

import cv2
import numpy as np

from ..types import Detection

# STRONGER QR DETECTOR
try:
    from pyzbar.pyzbar import decode as pyzbar_decode
    PYZBAR_AVAILABLE = True
except Exception:
    PYZBAR_AVAILABLE = False


class QRDetector:

    def __init__(self) -> None:
        self._detector = cv2.QRCodeDetector()

    def _normalize_points(self, pts: np.ndarray) -> np.ndarray:

        pts = np.asarray(pts, dtype=float)

        if pts.ndim == 3 and pts.shape[0] == 1:
            pts = pts[0]

        if pts.ndim == 3 and pts.shape[1] == 1:
            pts = pts.reshape(-1, 2)

        return pts

    def _make_detection(
        self,
        pts: np.ndarray,
        data: str,
        scale: float = 1.0,
        pad: int = 0,
    ) -> Detection:

        pts = self._normalize_points(pts)

        xs = (pts[:, 0] - pad) / scale
        ys = (pts[:, 1] - pad) / scale

        x1, y1 = int(xs.min()), int(ys.min())
        x2, y2 = int(xs.max()), int(ys.max())

        return Detection(
            label=f"qr:{data}" if data else "qr_code",
            confidence=1.0 if data else 0.7,
            bbox_xyxy=(x1, y1, x2, y2),
            source="qr",
            meta={"data": data or ""},
        )

    # =========================================
    # PYZBAR DETECTION
    # =========================================

    def _detect_pyzbar(
        self,
        image: np.ndarray,
        scale: float = 1.0,
        pad: int = 0,
    ) -> List[Detection]:

        detections = []

        if not PYZBAR_AVAILABLE:
            return detections

        try:

            decoded = pyzbar_decode(image)

            for obj in decoded:

                pts = np.array(
                    [
                        [obj.rect.left, obj.rect.top],
                        [obj.rect.left + obj.rect.width, obj.rect.top],
                        [obj.rect.left + obj.rect.width, obj.rect.top + obj.rect.height],
                        [obj.rect.left, obj.rect.top + obj.rect.height],
                    ],
                    dtype=np.float32,
                )

                text = obj.data.decode("utf-8")

                detections.append(
                    self._make_detection(
                        pts,
                        text,
                        scale=scale,
                        pad=pad,
                    )
                )

        except Exception:
            pass

        return detections

    # =========================================
    # OPENCV DETECTION
    # =========================================

    def _detect_opencv(
        self,
        image: np.ndarray,
        scale: float = 1.0,
        pad: int = 0,
    ) -> List[Detection]:

        detections = []

        try:

            data, points, _ = self._detector.detectAndDecode(image)

            if points is not None and len(points) > 0:

                detections.append(
                    self._make_detection(
                        points[0],
                        data,
                        scale=scale,
                        pad=pad,
                    )
                )

        except Exception:
            pass

        return detections

    # =========================================
    # PREPROCESSING
    # =========================================

    def _prepare(self, gray):

        # CLAHE improves uneven lighting
        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8),
        )

        gray = clahe.apply(gray)

        # Bilateral preserves QR edges
        gray = cv2.bilateralFilter(gray, 7, 50, 50)

        return gray

    # =========================================
    # RUN PASSES
    # =========================================

    def _run_passes(self, gray, scale):

        detections = []

        pad = 40

        gray = cv2.copyMakeBorder(
            gray,
            pad,
            pad,
            pad,
            pad,
            cv2.BORDER_CONSTANT,
            value=[255, 255, 255],
        )

        # PASS 1
        detections = self._detect_pyzbar(gray, scale, pad)

        if detections:
            return detections

        detections = self._detect_opencv(gray, scale, pad)

        if detections:
            return detections

        # PASS 2 - adaptive threshold
        try:

            binary = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                21,
                2,
            )

            detections = self._detect_pyzbar(binary, scale, pad)

            if detections:
                return detections

            detections = self._detect_opencv(binary, scale, pad)

            if detections:
                return detections

        except Exception:
            pass

        return []

    # =========================================
    # MAIN DETECTION
    # =========================================

    def detect(self, frame_bgr, upscale=False):

        if frame_bgr is None or frame_bgr.size == 0:
            return []

        # DOWNSCALE slightly to reduce Gemini noise
        frame_bgr = cv2.resize(
            frame_bgr,
            None,
            fx=0.75,
            fy=0.75,
            interpolation=cv2.INTER_AREA,
        )

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        gray = self._prepare(gray)

        detections = self._run_passes(gray, 1.0)

        if detections or not upscale:
            return detections

        # UPSCALE PASS
        h, w = gray.shape[:2]

        scale = 1.5

        up = cv2.resize(
            gray,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_LINEAR,
        )

        detections = self._run_passes(up, scale)

        return detections