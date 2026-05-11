from __future__ import annotations

from typing import Any, Dict, List


class WorldModel:
    """A shared representation of the current scene state."""

    def __init__(self) -> None:
        self.objects: Dict[int, Dict[str, Any]] = {}
        self.history: List[Dict[str, Any]] = []
        self.counts: Dict[str, int] = {}

    def update(self, detections: List[Dict[str, Any]]) -> None:
        for det in detections:
            object_id = det.get("id")
            if object_id is None:
                continue
            self.objects[object_id] = det.copy()

        self.history.append({"frame_objects": [det.copy() for det in detections]})
        self.counts = {}
        for det in detections:
            label = str(det.get("label", "unknown"))
            self.counts[label] = self.counts.get(label, 0) + 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "objects": {oid: data.copy() for oid, data in self.objects.items()},
            "counts": dict(self.counts),
            "history": list(self.history),
        }
