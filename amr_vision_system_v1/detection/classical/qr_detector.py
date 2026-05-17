from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

from ..types import Detection


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

    def _make_detection(self, pts: np.ndarray, data: str) -> Detection:
        pts = self._normalize_points(pts)
        if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 4:
            raise ValueError("Invalid QR points")

        xs = pts[:, 0]
        ys = pts[:, 1]
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        label = f"qr:{data}" if data else "qr_code"
        confidence = 1.0 if data else 0.6
        return Detection(
            label=label,
            confidence=confidence,
            bbox_xyxy=(x1, y1, x2, y2),
            source="qr",
            meta={"data": data or ""},
        )

    def _try_detect(self, image: np.ndarray) -> List[Detection]:
        """Try detection on a single image variant."""
        detections: List[Detection] = []
        try:
            if hasattr(self._detector, "detectAndDecodeMulti"):
                multi_result = self._detector.detectAndDecodeMulti(image)
                if not isinstance(multi_result, tuple):
                    return []

                decoded_info = []
                points = None

                if len(multi_result) == 3:
                    first, second, third = multi_result
                    if isinstance(first, bool):
                        decoded_info = second if isinstance(second, (list, tuple)) else []
                        points = third
                    else:
                        decoded_info = first if isinstance(first, (list, tuple)) else []
                        points = second
                elif len(multi_result) == 4:
                    first, second, third, _ = multi_result
                    if isinstance(first, bool):
                        decoded_info = second if isinstance(second, (list, tuple)) else []
                        points = third
                    else:
                        decoded_info = first if isinstance(first, (list, tuple)) else []
                        points = second
                else:
                    return []

                if points is None:
                    return []

                for index, pts in enumerate(points):
                    if pts is None:
                        continue
                    try:
                        detections.append(self._make_detection(pts, decoded_info[index] if index < len(decoded_info) and isinstance(decoded_info, (list, tuple)) else ""))
                    except ValueError:
                        continue
                return detections

            data, points, _ = self._detector.detectAndDecode(image)
            if points is None or len(points) == 0:
                return []
            try:
                detections.append(self._make_detection(points[0], data))
            except ValueError:
                pass

        except cv2.error:
            pass

        return detections

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        """Detect QR codes with multiple preprocessing strategies."""
        detections: List[Detection] = []
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # Try raw grayscale first
        detections.extend(self._try_detect(gray))
        if detections:
            return detections

        # Preprocessing: sharpening
        try:
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32)
            sharpened = cv2.filter2D(gray, -1, kernel)
            sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
            detections.extend(self._try_detect(sharpened))
            if detections:
                return detections
        except Exception:
            pass

        # Preprocessing: contrast enhancement (CLAHE)
        try:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            detections.extend(self._try_detect(enhanced))
            if detections:
                return detections
        except Exception:
            pass

        # Preprocessing: adaptive threshold
        try:
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            detections.extend(self._try_detect(binary))
            if detections:
                return detections
        except Exception:
            pass

        # Preprocessing: bilateral filter + adaptive threshold
        try:
            filtered = cv2.bilateralFilter(gray, 9, 75, 75)
            binary = cv2.adaptiveThreshold(
                filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            detections.extend(self._try_detect(binary))
        except Exception:
            pass

        return detections

