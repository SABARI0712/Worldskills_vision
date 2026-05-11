from __future__ import annotations

from typing import Any, Dict, List

from .duplicate_resolver import DuplicateResolver


class DetectionFuser:
    """Fuses detections from multiple sources into a cleaner set."""

    def __init__(self, iou_threshold: float = 0.45) -> None:
        self._resolver = DuplicateResolver(iou_threshold=iou_threshold)

    def fuse(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Currently fuse by resolving duplicates first.
        return self._resolver.resolve(detections)
