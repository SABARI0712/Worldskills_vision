from __future__ import annotations

from typing import Any, Dict, List, Optional

import cv2
import numpy as np


class Visualizer:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.line_thickness = int(cfg.get("line_thickness", 2))
        self.font_scale = float(cfg.get("font_scale", 0.5))

    def _draw_tag(self, image, pos, text, color):
        x, y = pos

        (w, h), baseline = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            self.font_scale,
            1
        )

        cv2.rectangle(
            image,
            (x, y - h - baseline - 4),
            (x + w + 4, y),
            color,
            -1
        )

        cv2.putText(
            image,
            text,
            (x + 2, y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    def draw(
        self,
        frame_bgr: np.ndarray,
        detections: List[Dict[str, Any]],
        grid: Optional[Dict[str, Any]] = None,
        occupancy: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:

        base = frame_bgr.copy()
        h, w = base.shape[:2]

        occupancy_layer = np.zeros_like(base)
        occupancy_label_layer = np.zeros_like(base)
        grid_layer = np.zeros_like(base)
        detection_layer = np.zeros_like(base)
        pose_layer = np.zeros_like(base)
        annotation_layer = np.zeros_like(base)

        # =========================================================
        # DRAW GRID + OCCUPANCY LAYERS
        # =========================================================

        if grid is not None:
            rows = int(grid.get("rows", 8))
            cols = int(grid.get("cols", 8))

            col_labels = grid.get("col_labels") or ""
            row_labels = grid.get("row_labels") or ""
            col_labels = str(col_labels)
            row_labels = str(row_labels)

            cell_w = w / cols
            cell_h = h / rows

            if occupancy is not None:
                occupancy_map = occupancy.get("occupancy_map", {})

                for cell, data in occupancy_map.items():
                    if len(cell) < 2:
                        continue

                    col_char = cell[0]
                    row_char = cell[1:]

                    try:
                        c = col_labels.index(col_char)
                        r = row_labels.index(row_char)
                    except ValueError:
                        continue

                    x1 = int(c * cell_w)
                    y1 = int(r * cell_h)
                    x2 = int((c + 1) * cell_w)
                    y2 = int((r + 1) * cell_h)

                    cv2.rectangle(
                        occupancy_layer,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 255),
                        -1
                    )

                    label = data.get("label") if isinstance(data, dict) else data
                    label = str(label)

                    cv2.putText(
                        occupancy_label_layer,
                        label,
                        (x1 + 5, y1 + 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 255, 255),
                        1,
                        cv2.LINE_AA
                    )

            for c in range(cols):
                x = int(c * cell_w)
                cv2.line(
                    grid_layer,
                    (x, 0),
                    (x, h),
                    (100, 100, 100),
                    1
                )

            for r in range(rows):
                y = int(r * cell_h)
                cv2.line(
                    grid_layer,
                    (0, y),
                    (w, y),
                    (100, 100, 100),
                    1
                )

            for r in range(rows):
                for c in range(cols):
                    if c >= len(col_labels) or r >= len(row_labels):
                        continue

                    cell_name = f"{col_labels[c]}{row_labels[r]}"
                    x = int(c * cell_w + 5)
                    y = int(r * cell_h + 20)

                    cv2.putText(
                        grid_layer,
                        cell_name,
                        (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (180, 180, 180),
                        1,
                        cv2.LINE_AA
                    )

        # =========================================================
        # DRAW DETECTIONS + POSE + ANNOTATIONS
        # =========================================================

        for det in detections:
            x1, y1, x2, y2 = map(int, det.get("bbox_xyxy", [0, 0, 0, 0]))
            label = str(det.get("label", "object"))
            source = str(det.get("source", ""))
            object_id = det.get("id")
            angle = det.get("angle")

            txt = label
            if object_id is not None:
                txt += f" #{object_id}"
            if angle is not None:
                txt += f" {angle:.1f}°"

            if source == "yolo":
                color = (255, 0, 0)
            elif source == "aruco":
                color = (0, 255, 0)
            elif source == "qr":
                color = (0, 255, 255)
            elif source == "ocr":
                color = (0, 215, 255)
            elif source == "color":
                color = (255, 255, 0)
            elif source == "contour":
                color = (0, 165, 255)
            else:
                color = (255, 255, 255)

            cv2.rectangle(
                detection_layer,
                (x1, y1),
                (x2, y2),
                color,
                self.line_thickness
            )

            self._draw_tag(
                annotation_layer,
                (x1, y1),
                txt,
                color
            )

            if angle is not None:
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                length = 50
                theta = np.deg2rad(angle)
                x_end = int(cx + length * np.cos(theta))
                y_end = int(cy - length * np.sin(theta))

                cv2.arrowedLine(
                    pose_layer,
                    (cx, cy),
                    (x_end, y_end),
                    (0, 0, 255),
                    2,
                    tipLength=0.25
                )

        out = base.copy()
        if occupancy is not None:
            out = cv2.addWeighted(out, 1.0, occupancy_layer, 0.25, 0)
            out = cv2.addWeighted(out, 1.0, occupancy_label_layer, 1.0, 0)
        if grid is not None:
            out = cv2.addWeighted(out, 1.0, grid_layer, 1.0, 0)
        out = cv2.addWeighted(out, 1.0, detection_layer, 1.0, 0)
        out = cv2.addWeighted(out, 1.0, pose_layer, 1.0, 0)
        out = cv2.addWeighted(out, 1.0, annotation_layer, 1.0, 0)

        return out

    def show(
        self,
        frame_bgr: np.ndarray,
        window_name: str = "AMR Vision"
    ) -> int:

        cv2.imshow(window_name, frame_bgr)
        return cv2.waitKey(1) & 0xFF

    def save(
        self,
        path: str,
        frame_bgr: np.ndarray
    ) -> None:

        cv2.imwrite(path, frame_bgr)