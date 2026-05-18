from __future__ import annotations

from typing import Any, Dict, List, Sequence


class DuplicateResolver:
    """Resolves duplicate or overlapping detections."""

    # Preferred specific colors beat less-specific ones when regions overlap.
    # Black is last because large dark/shadow regions frequently mimic objects.
    _COLOR_PRIORITY = ["red1", "red2", "red", "green", "blue", "orange", "yellow", "black"]

    def __init__(self, iou_threshold: float = 0.45) -> None:
        self.iou_threshold = iou_threshold
        self.source_priority = ["aruco", "qr", "ocr", "yolo", "color", "contour"]

    @staticmethod
    def _bbox_area(box: Sequence[float]) -> float:
        x1, y1, x2, y2 = box
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    @staticmethod
    def _intersection(box_a: Sequence[float], box_b: Sequence[float]) -> float:
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        return width * height

    def _iou(self, box_a: Sequence[float], box_b: Sequence[float]) -> float:
        inter = self._intersection(box_a, box_b)
        union = self._bbox_area(box_a) + self._bbox_area(box_b) - inter
        if union <= 0.0:
            return 0.0
        return inter / union

    def _priority(self, detection: Dict[str, Any]) -> int:
        source = str(detection.get("source", "")).lower()
        try:
            return self.source_priority.index(source)
        except ValueError:
            return len(self.source_priority)

    def _color_label_priority(self, label: str) -> int:
        """Lower index = higher priority colour."""
        color_name = label.replace("color:", "").lower()
        for i, c in enumerate(self._COLOR_PRIORITY):
            if c in color_name:
                return i
        return len(self._COLOR_PRIORITY)

    def resolve(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not detections:
            return []

        indexed = [dict(d, _index=i) for i, d in enumerate(detections)]
        indexed.sort(key=lambda item: (-float(item.get("confidence", 0.0)), self._priority(item), item["_index"]))

        selected: List[Dict[str, Any]] = []
        for det in indexed:
            bbox = det.get("bbox_xyxy", [0, 0, 0, 0])
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            det_label = str(det.get("label", "")).strip().lower()
            det_source = str(det.get("source", "")).strip().lower()
            suppress = False
            to_remove: List[Dict[str, Any]] = []
            for sel in selected:
                sel_bbox = sel.get("bbox_xyxy", [0, 0, 0, 0])
                if not isinstance(sel_bbox, list) or len(sel_bbox) != 4:
                    continue
                if self._iou(bbox, sel_bbox) <= self.iou_threshold:
                    continue
                sel_label = str(sel.get("label", "")).strip().lower()
                sel_source = str(sel.get("source", "")).strip().lower()
                if det_label == sel_label:
                    # Prefer detections from higher-priority sources regardless of confidence
                    det_prio = self._priority(det)
                    sel_prio = self._priority(sel)
                    if det_prio < sel_prio:
                        to_remove.append(sel)
                        continue
                    else:
                        suppress = True
                        break
                # Two colour detections overlapping with different labels:
                # keep the higher-priority (more specific) colour, suppress black/shadows.
                if det_source == "color" and sel_source == "color" and det_label != sel_label:
                    det_c_prio = self._color_label_priority(det_label)
                    sel_c_prio = self._color_label_priority(sel_label)
                    if det_c_prio <= sel_c_prio:
                        to_remove.append(sel)
                        continue
                    else:
                        suppress = True
                        break
                if det_source in ["qr", "aruco"] and sel_source == "color":
                    to_remove.append(sel)
                    continue
                if det_source == "color" and sel_source in ["qr", "aruco"]:
                    suppress = True
                    break
                if det_source == "yolo" and sel_source == "color":
                    to_remove.append(sel)
                    continue
                if det_source == "color" and sel_source == "yolo":
                    suppress = True
                    break
            if suppress:
                continue
            for sel in to_remove:
                if sel in selected:
                    selected.remove(sel)
            selected.append(det)

        selected.sort(key=lambda item: item["_index"])
        for item in selected:
            item.pop("_index", None)
        return selected
