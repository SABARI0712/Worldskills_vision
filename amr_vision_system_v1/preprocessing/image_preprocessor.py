from __future__ import annotations

from typing import Any, Dict, Tuple

import cv2
import numpy as np


class ImagePreprocessor:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg

        clahe_cfg = (cfg.get("clahe") or {}) if cfg else {}
        self._clahe = None
        if bool(clahe_cfg.get("enabled", False)):
            clip = float(clahe_cfg.get("clip_limit", 2.0))
            tgs = clahe_cfg.get("tile_grid_size", [8, 8])
            self._clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(int(tgs[0]), int(tgs[1])))

    def process(self, frame_bgr: np.ndarray) -> np.ndarray:
        out = frame_bgr

        resize_cfg = (self.cfg.get("resize") or {}) if self.cfg else {}
        if bool(resize_cfg.get("enabled", False)):
            w = int(resize_cfg.get("width", out.shape[1]))
            h = int(resize_cfg.get("height", out.shape[0]))
            keep_aspect = bool(resize_cfg.get("keep_aspect", False))
            out = self._resize(out, (w, h), keep_aspect)

        if self._clahe is not None:
            lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l2 = self._clahe.apply(l)
            out = cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2BGR)

        denoise_cfg = (self.cfg.get("denoise") or {}) if self.cfg else {}
        if bool(denoise_cfg.get("enabled", False)):
            out = cv2.fastNlMeansDenoisingColored(
                out,
                None,
                float(denoise_cfg.get("h", 5)),
                float(denoise_cfg.get("h_color", 5)),
                int(denoise_cfg.get("template_window_size", 7)),
                int(denoise_cfg.get("search_window_size", 21)),
            )

        return out

    @staticmethod
    def _resize(img: np.ndarray, size_wh: Tuple[int, int], keep_aspect: bool) -> np.ndarray:
        w, h = size_wh
        if not keep_aspect:
            return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        ih, iw = img.shape[:2]
        scale = min(w / iw, h / ih)
        nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
        return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

