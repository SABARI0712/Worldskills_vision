from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional


class OccupancyGrid:
    def __init__(self, allowed_sources: Optional[Iterable[str]] = None) -> None:
        self.allowed_sources = set(allowed_sources or ["yolo", "qr", "aruco"])

    def build(self, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        occupancy_map: Dict[str, Dict[str, Any]] = {}

        for det in detections:
            source = str(det.get("source", "")).lower()
            cell = det.get("cell")
            if not cell or source not in self.allowed_sources:
                continue

            confidence = float(det.get("confidence", 0.0)) if det.get("confidence") is not None else 0.0
            existing = occupancy_map.get(cell)
            if existing is None or confidence > float(existing.get("confidence", 0.0)):
                occupancy_map[cell] = {
                    "label": str(det.get("label", "object")),
                    "source": source,
                    "confidence": confidence,
                    "object_id": det.get("id"),
                    "color": det.get("color"),
                    "bbox_xyxy": det.get("bbox_xyxy"),
                }

        occupied_cells = sorted(occupancy_map.keys())
        return {"occupied_cells": occupied_cells, "occupancy_map": occupancy_map}
