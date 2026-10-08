from __future__ import annotations

from typing import Any, Dict, List, Optional

import cv2
import numpy as np


class Visualizer:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.line_thickness = int(cfg.get("line_thickness", 3))
        self.font_scale = float(cfg.get("font_scale", 0.75))
        self.tag_thickness = int(cfg.get("tag_thickness", 2))
        
        # Pre-rendered static layers for zero-latency drawing
        self._static_grid_overlay = None
        self._grid_params = None

    def _draw_tag(self, image, pos, text, color):
        x, y = pos

        (w, h), baseline = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            self.font_scale,
            self.tag_thickness
        )

        padding_x = 8
        padding_y = 6
        top_left = (x, max(0, y - h - baseline - padding_y))
        bottom_right = (x + w + padding_x, y)

        # Draw box and borders in-place
        cv2.rectangle(image, top_left, bottom_right, (0, 0, 0), -1)
        cv2.rectangle(image, top_left, bottom_right, color, 1)

        cv2.putText(
            image,
            text,
            (x + 4, y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.font_scale,
            (255, 255, 255),
            self.tag_thickness,
            cv2.LINE_AA
        )

    def draw(
        self,
        frame_bgr: np.ndarray,
        detections: List[Dict[str, Any]],
        grid: Optional[Dict[str, Any]] = None,
        occupancy: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:

        out = frame_bgr.copy()
        h, w = out.shape[:2]

        # =========================================================
        # 1. OPTIMIZATION: PRE-RENDER STATIC GRID
        # =========================================================
        if grid is not None:
            rows = int(grid.get("rows", 8))
            cols = int(grid.get("cols", 8))
            col_labels = str(grid.get("col_labels") or "")
            row_labels = str(grid.get("row_labels") or "")
            
            grid_params = (cols, rows, col_labels, row_labels)
            
            if (
                self._static_grid_overlay is None or
                self._static_grid_overlay.shape[:2] != (h, w) or
                self._grid_params != grid_params
            ):
                # Pre-render grid layers once onto a clean grid template
                self._static_grid_overlay = np.zeros_like(out)
                cell_w = w / cols
                cell_h = h / rows
                
                # Draw the basic grid lines
                for c in range(cols + 1):
                    x = int(c * cell_w)
                    cv2.line(self._static_grid_overlay, (x, 0), (x, h), (100, 100, 100), 1)
                for r in range(rows + 1):
                    y = int(r * cell_h)
                    cv2.line(self._static_grid_overlay, (0, y), (w, y), (100, 100, 100), 1)
                
                # Write cell names (A8, B7, etc.)
                for r in range(rows):
                    for c in range(cols):
                        if c < len(col_labels) and r < len(row_labels):
                            cell_name = f"{col_labels[c]}{row_labels[r]}"
                            x_pos = int(c * cell_w + 5)
                            y_pos = int(r * cell_h + 20)
                            cv2.putText(
                                self._static_grid_overlay,
                                cell_name,
                                (x_pos, y_pos),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.4,
                                (220, 220, 220),
                                1,
                                cv2.LINE_AA
                            )
                self._grid_params = grid_params
            
            # Apply grid overlay using fast addWeighted
            out = cv2.addWeighted(out, 1.0, self._static_grid_overlay, 1.0, 0)

        # =========================================================
        # 2. OPTIMIZATION: IN-PLACE OCCUPANCY ALPHABLEND
        # =========================================================
        if grid is not None and occupancy is not None:
            rows = int(grid.get("rows", 8))
            cols = int(grid.get("cols", 8))
            col_labels = str(grid.get("col_labels") or "")
            row_labels = str(grid.get("row_labels") or "")
            cell_w = w / cols
            cell_h = h / rows
            
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

                # Localized sub-image transparent blending
                sub_img = out[y1:y2, x1:x2]
                rect = np.full(sub_img.shape, (0, 255, 255), dtype=np.uint8)
                out[y1:y2, x1:x2] = cv2.addWeighted(sub_img, 0.75, rect, 0.25, 0)

                # Cell label on top
                label = data.get("label") if isinstance(data, dict) else data
                label = str(label)
                cv2.putText(
                    out,
                    label,
                    (x1 + 5, y1 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA
                )

        # =========================================================
        # 3. OPTIMIZATION: IN-PLACE DETECTIONS (WITH Z-ORDER SORTING)
        # =========================================================
        # Sort so highest priority (QR/ArUco/OCR) are rendered last (on top of others)
        priority_map = {"qr": 3, "aruco": 2, "ocr": 2, "barcode": 2, "yolo": 1, "color": 0, "contour": 0}
        sorted_detections = sorted(
            detections,
            key=lambda d: priority_map.get(str(d.get("source", "")).lower(), 0)
        )

        for det in sorted_detections:
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

            color_map = {
                "yolo": (255, 0, 0),
                "aruco": (0, 255, 0),
                "qr": (0, 255, 255),
                "ocr": (0, 215, 255),
                "barcode": (255, 0, 255),
                "color": (255, 255, 0),
                "contour": (0, 165, 255),
            }
            color = color_map.get(source, (255, 255, 255))

            # Bounding box
            cv2.rectangle(out, (x1, y1), (x2, y2), color, self.line_thickness)
            
            # Centroid
            centroid = det.get("centroid")
            if centroid and isinstance(centroid, (list, tuple)) and len(centroid) == 2:
                cx, cy = int(centroid[0]), int(centroid[1])
            else:
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

            cv2.circle(out, (cx, cy), 5, color, -1)
            cv2.line(out, (cx - 8, cy), (cx + 8, cy), color, 1)
            cv2.line(out, (cx, cy - 8), (cx, cy + 8), color, 1)

            # Metadata/Velocity vector
            velocity = None
            meta = det.get("meta") or {}
            if isinstance(meta, dict):
                velocity = meta.get("velocity")
            if velocity:
                vel_text = f"v={velocity[0]},{velocity[1]}"
                self._draw_tag(out, (x1, y2 + 20), vel_text, color)

            # Text tag
            self._draw_tag(out, (x1, y1), txt, color)

            # Angle direction vector
            if angle is not None:
                length = 50
                theta = np.deg2rad(angle)
                x_end = int(cx + length * np.cos(theta))
                y_end = int(cy - length * np.sin(theta))

                cv2.arrowedLine(
                    out,
                    (cx, cy),
                    (x_end, y_end),
                    (0, 0, 255),
                    2,
                    tipLength=0.25
                )

        # Clean borders for chess cells/grid overlay
        if grid is not None:
            cell_w = w / cols
            cell_h = h / rows
            for c in range(cols + 1):
                x = int(c * cell_w)
                cv2.line(out, (x, 0), (x, h), (230, 230, 230), 2)
            for r in range(rows + 1):
                y = int(r * cell_h)
                cv2.line(out, (0, y), (w, y), (230, 230, 230), 2)

        return out

    def draw_counts(self, frame: np.ndarray, count_summary: Dict[str, Any], position=(10, 30)) -> np.ndarray:
        img = frame.copy()
        y = position[1]
        h = 22

        cv2.putText(
            img,
            "=== COUNT SUMMARY ===",
            (position[0], y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )
        y += h

        total = count_summary.get("total_objects", 0)
        cv2.putText(
            img,
            f"Total: {total}",
            (position[0], y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
        y += h

        by_label = count_summary.get("by_label", {})
        for label, cnt in list(by_label.items())[:10]:
            cv2.putText(
                img,
                f"  {label}: {cnt}",
                (position[0], y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )
            y += h

        return img

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