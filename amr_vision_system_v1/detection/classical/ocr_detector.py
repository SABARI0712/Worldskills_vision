from __future__ import annotations

import cv2
import numpy as np
import time
from typing import List, Dict, Any, Optional

from ..types import Detection


class OCRDetector:
    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        self.cfg = cfg or {}
        
        # Base logic settings derived from ocr_working_live.py:
        # Default to low confidence, small text, small area to let the full live OCR
        # detections flow into the pipeline, while also matching config overrides if present.
        self.min_conf = float(self.cfg.get("min_confidence", 0.10))
        self.min_text_len = int(self.cfg.get("min_text_length", 2))
        self.min_area = int(self.cfg.get("min_area", 10))
        self.max_aspect_ratio = float(self.cfg.get("max_aspect_ratio", 20.0))
        self._tesseract_config = str(self.cfg.get("tesseract_config", "--psm 11"))
        
        self.scan_counter = 0

        self.tesseract: Any = None
        try:
            import pytesseract
            self.tesseract = pytesseract
        except ImportError:
            print("❌ pytesseract not installed. OCR disabled.")

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        if self.tesseract is None or frame_bgr is None or frame_bgr.size == 0:
            return []

        self.scan_counter += 1

        # ─────────────────────────────────────────────────────────────
        # BASE LOGIC: Grayscale conversion (matching ocr_working_live.py)
        # ─────────────────────────────────────────────────────────────
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # ─────────────────────────────────────────────────────────────
        # PIPELINE INTEGRATION: Get bounding boxes for the AMR pipeline
        # ─────────────────────────────────────────────────────────────
        dets: List[Detection] = []
        try:
            t0 = time.time()
            data = self.tesseract.image_to_data(
                gray,
                output_type=self.tesseract.Output.DICT,
                config=self._tesseract_config
            )
            elapsed = time.time() - t0
            n_boxes = len(data.get('text', []))

            for i in range(n_boxes):
                word_text = str(data['text'][i]).strip()
                try:
                    conf = float(data['conf'][i])
                except (TypeError, ValueError):
                    continue

                if not word_text or conf < (self.min_conf * 100) or len(word_text) < self.min_text_len:
                    continue

                x = int(data['left'][i])
                y = int(data['top'][i])
                w = int(data['width'][i])
                h = int(data['height'][i])

                if w <= 0 or h <= 0:
                    continue

                area = w * h
                if area < self.min_area or w < 5 or h < 5:
                    continue

                aspect = max(w / h, h / w)
                if aspect > self.max_aspect_ratio:
                    continue

                dets.append(
                    Detection(
                        label=f"ocr:{word_text}",
                        confidence=round(conf / 100, 2),
                        bbox_xyxy=(x, y, x + w, y + h),
                        source="ocr",
                        meta={"text": word_text, "confidence": round(conf / 100, 2)},
                    )
                )
        except Exception:
            pass

        return dets
