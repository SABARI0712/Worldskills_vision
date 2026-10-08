from __future__ import annotations

from typing import Any, Dict, List, Optional


class WorldModel:
    """A shared representation of the current scene state.

    ``objects`` holds the latest detection dict for every tracked ID.
    ``history`` is a capped sliding window of recent frames.
    ``counts`` is the per-label count for the most recent frame.
    """

    def __init__(self, max_history: int = 200) -> None:
        self.max_history = max_history
        self.objects: Dict[int, Dict[str, Any]] = {}
        self.history: List[Dict[str, Any]] = []
        self.counts: Dict[str, int] = {}

    def update(self, detections: List[Dict[str, Any]], alive_ids: Optional[set] = None) -> None:
        for det in detections:
            object_id = det.get("id")
            if object_id is None:
                continue
            self.objects[object_id] = det.copy()

        # Remove objects the tracker has already deregistered
        if alive_ids is not None:
            for oid in [k for k in self.objects if k not in alive_ids]:
                self.objects.pop(oid)

        self.history.append({"frame_objects": [det.copy() for det in detections]})
        # Trim history to stay within cap
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

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
