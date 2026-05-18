from __future__ import annotations

import cv2
import numpy as np
from typing import List, Dict, Any

from ..types import Detection


class OCRDetector:
    def __init__(self, cfg: Dict[str, Any] = None):
        self.cfg = cfg or {}

        # min_conf is stored as 0-1 (e.g. 0.75).
        # Tesseract returns 0-100; we multiply by 100 before comparing.
        self.min_conf = float(self.cfg.get("min_confidence", 0.75))
        self.min_text_len = int(self.cfg.get("min_text_length", 3))
        self.min_area = int(self.cfg.get("min_area", 1500))
        self.max_aspect_ratio = float(self.cfg.get("max_aspect_ratio", 5.0))

        # PSM 6 = uniform block of text (good for labels)
        # PSM 11 = sparse text (good for industrial environments with scattered text)
        wl = r'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 -_.'
        self._cfg_psm6 = f'--oem 3 --psm 6 -c tessedit_char_whitelist="{wl}"'
        self._cfg_psm11 = f'--oem 3 --psm 11 -c tessedit_char_whitelist="{wl}"'

        try:
            import pytesseract
            self.tesseract = pytesseract
        except ImportError:
            print("❌ pytesseract not installed. OCR disabled.")
            self.tesseract = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _try_detect(self, image: np.ndarray, scale: float = 1.0,
                    tsconfig: str | None = None) -> List[Detection]:
        """Run tesseract on *image* and return Detection list.

        *scale* is the factor by which the image was upscaled relative to the
        original frame; bounding boxes are divided by scale so they map back to
        the original coordinate space.
        """
        if tsconfig is None:
            tsconfig = self._cfg_psm6
        try:
            data = self.tesseract.image_to_data(
                image,
                output_type=self.tesseract.Output.DICT,
                config=tsconfig,
            )
        except Exception:
            return []

        dets: List[Detection] = []
        n_boxes = len(data.get("text", []))
        # Threshold in 0-100 scale to match tesseract output
        thresh = self.min_conf * 100.0

        for i in range(n_boxes):
            text = str(data["text"][i]).strip()
            try:
                conf = float(data["conf"][i])
            except (TypeError, ValueError):
                continue

            # Apply correct threshold (tesseract conf is 0-100)
            if not text or conf < thresh or len(text) < self.min_text_len:
                continue

            # Scale coordinates back to original image space
            x = int(data["left"][i] / scale)
            y = int(data["top"][i] / scale)
            w = int(data["width"][i] / scale)
            h = int(data["height"][i] / scale)

            if w <= 0 or h <= 0 or w < 20 or h < 15:
                continue

            area = w * h
            if area < self.min_area:
                continue

            aspect = max(w / h, h / w)
            if aspect > self.max_aspect_ratio:
                continue

            dets.append(
                Detection(
                    label=f"ocr:{text}",
                    confidence=round(conf / 100.0, 3),
                    bbox_xyxy=(x, y, x + w, y + h),
                    source="ocr",
                    meta={"text": text, "raw_conf": conf},
                )
            )

        return dets

    @staticmethod
    def _iou(a, b) -> float:
        x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
        x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
        area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))
        return inter / (area_a + area_b - inter)

    def _dedup(self, detections: List[Detection]) -> List[Detection]:
        """Remove duplicates: same label text at overlapping bbox → keep highest conf."""
        seen: List[Detection] = []
        for det in sorted(detections, key=lambda d: -d.confidence):
            is_dup = False
            for s in seen:
                if s.label == det.label and self._iou(s.bbox_xyxy, det.bbox_xyxy) > 0.3:
                    is_dup = True
                    break
            if not is_dup:
                seen.append(det)
        return seen

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        if self.tesseract is None or frame_bgr is None or frame_bgr.size == 0:
            return []

        h, w = frame_bgr.shape[:2]

        # ── Upscale for better OCR accuracy ────────────────────────────
        # Tesseract needs approximately 300 DPI. Most robot cameras deliver
        # text that is far too small at native resolution.  2x cubic
        # interpolation is the single biggest accuracy improvement.
        scale = 2.0 if w < 1000 else 1.5
        up = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)),
                        interpolation=cv2.INTER_CUBIC)

        # ── Base grayscale + CLAHE ──────────────────────────────────────
        gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        all_dets: List[Detection] = []

        # Pass 1 — CLAHE-enhanced, PSM 6
        all_dets.extend(self._try_detect(gray, scale, self._cfg_psm6))
        if all_dets:
            return self._dedup(all_dets)

        # Pass 2 — Sharpen + CLAHE, PSM 6
        try:
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]],
                              dtype=np.float32)
            sharpened = np.clip(cv2.filter2D(gray, -1, kernel), 0, 255).astype(np.uint8)
            all_dets.extend(self._try_detect(sharpened, scale, self._cfg_psm6))
            if all_dets:
                return self._dedup(all_dets)
        except Exception:
            pass

        # Pass 3 — Adaptive threshold, PSM 6
        try:
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2,
            )
            all_dets.extend(self._try_detect(binary, scale, self._cfg_psm6))
            if all_dets:
                return self._dedup(all_dets)
        except Exception:
            pass

        # Pass 4 — Bilateral + adaptive threshold, PSM 11 (sparse text)
        try:
            filtered = cv2.bilateralFilter(gray, 9, 75, 75)
            binary = cv2.adaptiveThreshold(
                filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2,
            )
            all_dets.extend(self._try_detect(binary, scale, self._cfg_psm11))
        except Exception:
            pass

        return self._dedup(all_dets)
