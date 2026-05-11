from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


class Visualizer:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.window_name = str(cfg.get("window_name", "AMR Vision V1"))
        self.draw_grid = bool(cfg.get("draw_grid", True))
        self.draw_labels = bool(cfg.get("draw_labels", True))
        self.draw_confidence = bool(cfg.get("draw_confidence", True))

    def draw(self, frame_bgr: np.ndarray, dets: List[Dict[str, Any]], grid: Optional[Dict[str, Any]] = None) -> np.ndarray:
        out = frame_bgr.copy()
        h, w = out.shape[:2]

        if self.draw_grid and grid:
            self._draw_grid(out, grid)

        for d in dets:
            bbox = d.get("bbox_xyxy", None)
            if not bbox or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = map(int, bbox)
            src = str(d.get("source", "det"))
            color = (0, 255, 0) if src == "yolo" else (255, 200, 0)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            if self.draw_labels:
                label = str(d.get("label", "obj"))
                conf = float(d.get("confidence", 0.0))
                cell = d.get("cell", None)
                txt = label
                if self.draw_confidence:
                    txt += f" {conf:.2f}"
                if cell:
                    txt += f" {cell}"
                self._draw_tag(out, (x1, y1), txt, color)

        return out

    def show(self, image_bgr: np.ndarray) -> int:
        cv2.imshow(self.window_name, image_bgr)
        return cv2.waitKey(1) & 0xFF

    def save(self, path: str, image_bgr: np.ndarray) -> None:
        import os

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        cv2.imwrite(path, image_bgr)

    @staticmethod
    def _draw_tag(img: np.ndarray, xy: Tuple[int, int], text: str, color: Tuple[int, int, int]) -> None:
        x, y = xy
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        y0 = max(0, y - th - 10)
        cv2.rectangle(img, (x, y0), (x + tw + 10, y0 + th + 10), color, -1)
        cv2.putText(img, text, (x + 5, y0 + th + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

    @staticmethod
    def _draw_grid(img: np.ndarray, grid: Dict[str, Any]) -> None:
        rows = int(grid.get("rows", 8))
        cols = int(grid.get("cols", 8))
        col_labels = str(grid.get("col_labels", "ABCDEFGH"))
        row_labels = str(grid.get("row_labels", "87654321"))
        h, w = img.shape[:2]

        for c in range(1, cols):
            x = int(w * c / cols)
            cv2.line(img, (x, 0), (x, h), (180, 180, 180), 1)
        for r in range(1, rows):
            y = int(h * r / rows)
            cv2.line(img, (0, y), (w, y), (180, 180, 180), 1)

        for c in range(min(cols, len(col_labels))):
            x = int(w * (c + 0.02) / cols)
            cv2.putText(img, col_labels[c], (x, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2)
        for r in range(min(rows, len(row_labels))):
            y = int(h * (r + 0.15) / rows)
            cv2.putText(img, row_labels[r], (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2)

