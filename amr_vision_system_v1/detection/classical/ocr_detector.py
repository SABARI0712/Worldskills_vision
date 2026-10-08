from __future__ import annotations

import cv2
import numpy as np
from typing import List, Dict, Any, Optional

from ..types import Detection


class OCRDetector:
    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        self.cfg = cfg or {}

        self.min_conf = float(self.cfg.get("min_confidence", 0.10))
        self.min_text_len = int(self.cfg.get("min_text_length", 1))
        self.min_area = int(self.cfg.get("min_area", 10))
        self.max_aspect_ratio = float(self.cfg.get("max_aspect_ratio", 20.0))
        self._tesseract_config = str(self.cfg.get("tesseract_config", "--oem 3 --psm 6"))

        self.tesseract: Any = None
        try:
            import pytesseract
            self.tesseract = pytesseract
        except Exception:
            self.tesseract = None

    def _parse_data(self, data: Dict[str, Any]) -> List[Detection]:
        dets: List[Detection] = []
        n_boxes = len(data.get("text", []))
        for i in range(n_boxes):
            word_text = str(data.get("text", [""])[i]).strip()
            try:
                conf = float(data.get("conf", ["-1"])[i])
            except Exception:
                continue

            if not word_text or conf < (self.min_conf * 100) or len(word_text) < self.min_text_len:
                continue

            x = int(data.get("left", [0])[i])
            y = int(data.get("top", [0])[i])
            w = int(data.get("width", [0])[i])
            h = int(data.get("height", [0])[i])

            if w <= 0 or h <= 0:
                continue

            area = w * h
            if area < self.min_area:
                continue

            aspect = max(w / h, h / w)
            if aspect > self.max_aspect_ratio:
                continue

            dets.append(
                Detection(
                    label=f"ocr:{word_text}",
                    confidence=round(conf / 100.0, 2),
                    bbox_xyxy=(x, y, x + w, y + h),
                    source="ocr",
                    meta={"text": word_text, "confidence": round(conf / 100.0, 2)},
                )
            )

        return dets

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        if self.tesseract is None or frame_bgr is None or frame_bgr.size == 0:
            return []

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        kernel = np.array([
            [-1, -1, -1],
            [-1, 9, -1],
            [-1, -1, -1],
        ], dtype=np.float32)
        gray = cv2.filter2D(gray, -1, kernel)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        pre_clahe = gray.copy()
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(pre_clahe)

        binary = cv2.adaptiveThreshold(
            enhanced,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2,
        )

        variants = [pre_clahe, enhanced, binary]
        try:
            bilateral = cv2.bilateralFilter(pre_clahe, 9, 75, 75)
            variants.append(bilateral)
        except Exception:
            pass

        all_detections: List[Detection] = []
        for img in variants:
            try:
                data = self.tesseract.image_to_data(img, output_type=self.tesseract.Output.DICT, config=self._tesseract_config)
                dets = self._parse_data(data)
                if dets:
                    all_detections.extend(dets)
            except Exception:
                continue

        if all_detections:
            all_detections.sort(key=lambda d: d.confidence, reverse=True)
            return all_detections

        return []
