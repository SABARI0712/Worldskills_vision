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

        try:
            import pytesseract
            self.tesseract = pytesseract
        except ImportError:
            print("❌ pytesseract not installed. OCR disabled.")
            self.tesseract = None

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        if self.tesseract is None or frame_bgr is None or frame_bgr.size == 0:
            return []

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        try:
            data = self.tesseract.image_to_data(gray, output_type=self.tesseract.Output.DICT)
        except Exception:
            return []

        dets: List[Detection] = []
        n_boxes = len(data['text'])

        for i in range(n_boxes):
            text = str(data['text'][i]).strip()
            conf = float(data['conf'][i])

            # Strong filtering
            if not text or conf < self.min_conf or len(text) < self.min_text_len:
                continue

            x = int(data['left'][i])
            y = int(data['top'][i])
            w = int(data['width'][i])
            h = int(data['height'][i])

            area = w * h
            if area < self.min_area or w == 0 or h == 0:
                continue

            # Reject very thin or very tall boxes (typical noise)
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
