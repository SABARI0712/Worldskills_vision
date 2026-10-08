from __future__ import annotations

from typing import Any, Dict, List


class SceneMemory:
    """Stores recent perception frames and scene changes."""

    def __init__(self, max_history: int = 100) -> None:
        self.max_history = int(max_history)
        self.frames: List[Dict[str, Any]] = []

    def record_frame(self, frame_id: int, detections: List[Dict[str, Any]]) -> None:
        self.frames.append({"frame_id": frame_id, "detections": [det.copy() for det in detections]})
        if len(self.frames) > self.max_history:
            self.frames.pop(0)

    def get_history(self) -> List[Dict[str, Any]]:
        return list(self.frames)
