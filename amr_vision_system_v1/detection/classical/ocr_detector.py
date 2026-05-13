from __future__ import annotations

from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import warnings

from ..types import Detection


class OCRDetector:
    def __init__(self, cfg: Dict[str, Any] | None = None) -> None:
        cfg = cfg or {}
        self.enabled = bool(cfg.get("enabled", True))
        self.languages = [str(lang) for lang in cfg.get("languages", ["en"])]
        self.confidence_threshold = float(cfg.get("confidence_threshold", 0.6))
        self.use_clahe = bool(cfg.get("use_clahe", True))
        self.clahe_clip = float(cfg.get("clahe_clip_limit", 2.0))
        self.clahe_grid = tuple(cfg.get("clahe_tile_grid_size", [8, 8]))
        self.use_mser = bool(cfg.get("use_mser", True))
        self.max_regions = int(cfg.get("max_regions", 20))
        self.min_region_area = int(cfg.get("min_region_area", 100))
        self.max_region_area = int(cfg.get("max_region_area", 40000))
        self.padding = int(cfg.get("padding", 8))
        self.tesseract_config = str(cfg.get("tesseract_config", "--oem 3 --psm 6"))
        self.lang = "+".join(self._normalize_languages(self.languages))
        self.tesseract = self._make_tesseract()

    def _normalize_languages(self, languages: List[str]) -> List[str]:
        mapping = {
            "en": "eng",
            "eng": "eng",
        }
        normalized: List[str] = []
        for lang in languages:
            lang_code = str(lang).strip().lower()
            normalized.append(mapping.get(lang_code, lang_code))
        return normalized

    def _make_tesseract(self):
        try:
            import pytesseract
        except ImportError:
            warnings.warn(
                "pytesseract is not installed. OCRDetector will be disabled. "
                "Install pytesseract and the Tesseract binary to enable OCR support."
            )
            return None

        try:
            pytesseract.get_tesseract_version()
            return pytesseract
        except Exception as exc:
            warnings.warn(f"Tesseract is not available or not configured correctly: {exc}")
            return None

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        if not self.enabled or self.tesseract is None:
            return []

        if frame_bgr is None or frame_bgr.size == 0:
            return []

        height, width = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        if self.use_clahe:
            clahe = cv2.createCLAHE(
                clipLimit=self.clahe_clip,
                tileGridSize=self.clahe_grid,
            )
            gray = clahe.apply(gray)

        if self.use_mser:
            rois = self._find_text_regions(gray)
        else:
            rois = [(0, 0, width, height)]

        if not rois:
            rois = [(0, 0, width, height)]

        detections: List[Detection] = []
        for x1, y1, x2, y2 in rois:
            x1, y1, x2, y2 = self._clamp_roi(x1, y1, x2, y2, width, height)
            x1 = max(0, x1 - self.padding)
            y1 = max(0, y1 - self.padding)
            x2 = min(width, x2 + self.padding)
            y2 = min(height, y2 + self.padding)

            roi = frame_bgr[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            roi_gray = self._prepare_roi(roi_gray)

            try:
                data = self.tesseract.image_to_data(
                    roi_gray,
                    lang=self.lang,
                    config=self.tesseract_config,
                    output_type=self.tesseract.Output.DICT,
                )
            except Exception:
                continue

            for i, text in enumerate(data.get("text", [])):
                if text is None:
                    continue
                text = str(text).strip()
                if not text:
                    continue

                try:
                    conf = float(data.get("conf", [])[i])
                except Exception:
                    continue

                if conf < self.confidence_threshold * 100.0:
                    continue

                try:
                    rx1 = int(data.get("left", [])[i])
                    ry1 = int(data.get("top", [])[i])
                    rw = int(data.get("width", [])[i])
                    rh = int(data.get("height", [])[i])
                except Exception:
                    continue

                if rw <= 0 or rh <= 0:
                    continue

                rxx1 = x1 + rx1
                ryy1 = y1 + ry1
                rxx2 = rxx1 + rw
                ryy2 = ryy1 + rh

                detections.append(
                    Detection(
                        label=f"ocr:{text}",
                        confidence=conf / 100.0,
                        bbox_xyxy=(rxx1, ryy1, rxx2, ryy2),
                        source="ocr",
                        meta={"text": text, "confidence": conf / 100.0},
                    )
                )

        return detections

    def _prepare_roi(self, gray: np.ndarray) -> np.ndarray:
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        return thresh

    def _find_text_regions(self, gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
        regions: List[Tuple[int, int, int, int]] = []
        try:
            mser = cv2.MSER_create()
            mser_regions, _ = mser.detectRegions(gray)
        except Exception:
            return []

        for region in mser_regions:
            x, y, w, h = cv2.boundingRect(region.reshape(-1, 1, 2))
            area = w * h
            if area < self.min_region_area or area > self.max_region_area:
                continue
            aspect = float(w) / float(max(h, 1))
            if aspect < 0.2 or aspect > 10.0:
                continue
            regions.append((x, y, x + w, y + h))

        regions = self._merge_regions(regions)
        regions = sorted(regions, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]), reverse=True)
        return regions[: self.max_regions]

    def _clamp_roi(self, x1: int, y1: int, x2: int, y2: int, max_w: int, max_h: int) -> Tuple[int, int, int, int]:
        x1 = max(0, min(x1, max_w))
        y1 = max(0, min(y1, max_h))
        x2 = max(0, min(x2, max_w))
        y2 = max(0, min(y2, max_h))
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        return x1, y1, x2, y2

    @staticmethod
    def _merge_regions(regions: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        merged: List[Tuple[int, int, int, int]] = []

        def intersects(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> int:
            x1 = max(a[0], b[0])
            y1 = max(a[1], b[1])
            x2 = min(a[2], b[2])
            y2 = min(a[3], b[3])
            if x2 <= x1 or y2 <= y1:
                return 0
            return (x2 - x1) * (y2 - y1)

        for region in regions:
            merged_flag = False
            for idx, existing in enumerate(merged):
                inter = intersects(region, existing)
                if inter > 0:
                    x1 = min(region[0], existing[0])
                    y1 = min(region[1], existing[1])
                    x2 = max(region[2], existing[2])
                    y2 = max(region[3], existing[3])
                    merged[idx] = (x1, y1, x2, y2)
                    merged_flag = True
                    break
            if not merged_flag:
                merged.append(region)

        return merged
