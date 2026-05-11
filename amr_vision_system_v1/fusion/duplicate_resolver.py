from __future__ import annotations

from typing import Any, Dict, List, Sequence


class DuplicateResolver:
    """Resolves duplicate or overlapping detections."""

    def __init__(self, iou_threshold: float = 0.45) -> None:
        self.iou_threshold = float(iou_threshold)
        self.source_priority = ["yolo", "qr", "color", "contour"]

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

    def resolve(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not detections:
            return []

        indexed = [dict(d, _index=i) for i, d in enumerate(detections)]
        indexed.sort(key=lambda item: (-float(item.get("confidence", 0.0)), self._priority(item), item["_index"]))

        selected: List[Dict[str, Any]] = []
        for det in indexed:
            bbox = det.get("bbox_xyxy", [0, 0, 0, 0])
            if len(bbox) != 4:
                continue
            if any(self._iou(bbox, sel.get("bbox_xyxy", [0, 0, 0, 0])) > self.iou_threshold for sel in selected):
                continue
            selected.append(det)

        selected.sort(key=lambda item: item["_index"])
        for item in selected:
            item.pop("_index", None)
        return selected
