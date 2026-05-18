from __future__ import annotations

import cv2
import numpy as np
from typing import List, Dict, Any

from ..types import Detection


class OCRDetector:
    def __init__(self, cfg: Dict[str, Any] = None):
        self.cfg = cfg or {}
        
        self.min_conf = float(self.cfg.get("min_confidence", 0.85))
        self.min_text_len = int(self.cfg.get("min_text_length", 3))
        self.min_area = int(self.cfg.get("min_area", 1500))
        self.max_aspect_ratio = float(self.cfg.get("max_aspect_ratio", 5.0))
        self._tesseract_config = str(self.cfg.get("tesseract_config", '--oem 3 --psm 6 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 "'))

        try:
            import pytesseract
            self.tesseract = pytesseract
        except ImportError:
            print("❌ pytesseract not installed. OCR disabled.")
            self.tesseract = None

    def _try_detect(self, image: np.ndarray) -> List[Detection]:
        try:
            data = self.tesseract.image_to_data(image, output_type=self.tesseract.Output.DICT, config=self._tesseract_config)
        except Exception:
            return []

        dets: List[Detection] = []
        n_boxes = len(data.get('text', []))

        for i in range(n_boxes):
            text = str(data['text'][i]).strip()
            try:
                conf = float(data['conf'][i])
            except (TypeError, ValueError):
                continue

            if not text or conf < self.min_conf or len(text) < self.min_text_len:
                continue

            x = int(data['left'][i])
            y = int(data['top'][i])
            w = int(data['width'][i])
            h = int(data['height'][i])

            if w <= 0 or h <= 0:
                continue

            area = w * h
            if area < self.min_area or w < 20 or h < 15:
                continue

            aspect = max(w / h, h / w)
            if aspect > self.max_aspect_ratio:
                continue

            dets.append(
                Detection(
                    label=f"ocr:{text}",
                    confidence=round(conf / 100, 2),
                    bbox_xyxy=(x, y, x + w, y + h),
                    source="ocr",
                    meta={"text": text, "confidence": round(conf / 100, 2)},
                )
            )

        return dets

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        if self.tesseract is None or frame_bgr is None or frame_bgr.size == 0:
            return []

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        # Sharpen and denoise
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32)
        gray = cv2.filter2D(gray, -1, kernel)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        detections = self._try_detect(gray)
        if detections:
            return detections

        try:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            detections = self._try_detect(enhanced)
            if detections:
                return detections
        except Exception:
            pass

        try:
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            detections = self._try_detect(binary)
            if detections:
                return detections
        except Exception:
            pass

        try:
            filtered = cv2.bilateralFilter(gray, 9, 75, 75)
            binary = cv2.adaptiveThreshold(
                filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            detections = self._try_detect(binary)
        except Exception:
            pass

        return detections
