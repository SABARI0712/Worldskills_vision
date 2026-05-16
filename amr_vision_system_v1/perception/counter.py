from collections import defaultdict
from typing import Dict, List, Any


class ObjectCounter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.total_by_label = defaultdict(int)
        self.by_cell = defaultdict(lambda: defaultdict(int))
        self.by_color = defaultdict(int)
        self.tracked_ids = set()

    def update(self, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Update counts for the current frame only.

        This method returns a per-frame snapshot rather than cumulative totals.
        """
        self.reset()

        for det in detections:
            label = str(det.get("label", "unknown")).replace("ocr:", "").replace("yolo:", "")
            cell = det.get("cell")
            color = det.get("color")
            obj_id = det.get("id")

            self.total_by_label[label] += 1

            if cell:
                self.by_cell[cell][label] += 1
            if color:
                self.by_color[f"{label}_{color}"] += 1
            if obj_id is not None:
                self.tracked_ids.add(obj_id)

        return self.get_summary()

    def get_summary(self) -> Dict[str, Any]:
        return {
            "total_objects": sum(self.total_by_label.values()),
            "by_label": dict(self.total_by_label),
            "by_cell": {k: dict(v) for k, v in self.by_cell.items() if v},
            "by_color": dict(self.by_color),
            "unique_tracked": len(self.tracked_ids)
        }
