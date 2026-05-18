from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, List, Tuple
import time

import numpy as np


class CentroidTracker:
    def __init__(self, max_disappeared: int = 8, max_distance: int = 80) -> None:
        self.next_object_id = 1
        self.objects: OrderedDict[int, Tuple[int, int]] = OrderedDict()
        self.labels: OrderedDict[int, str] = OrderedDict()
        self.disappeared: OrderedDict[int, int] = OrderedDict()
        self.velocities: OrderedDict[int, Tuple[float, float]] = OrderedDict()
        self.last_update_time = time.time()

        self.max_disappeared = max_disappeared
        self.max_distance = float(max_distance)

    def update(self, detections: List[Dict[str, Any]], timestamp_ms: int | None = None) -> List[Dict[str, Any]]:
        # Use provided frame timestamp when available (milliseconds), else fallback to wall clock
        now = (timestamp_ms / 1000.0) if timestamp_ms is not None else time.time()
        dt = now - self.last_update_time
        if dt <= 0:
            dt = 0.033
        self.last_update_time = now

        if detections is None:
            detections = []

        if len(detections) == 0:
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self._deregister(oid)
            return []

        input_centroids: List[Tuple[int, int, Dict[str, Any]]] = []
        for det in detections:
            x1, y1, x2, y2 = map(int, det.get("bbox_xyxy", [0, 0, 0, 0]))
            cX = int((x1 + x2) / 2.0)
            cY = int((y1 + y2) / 2.0)
            input_centroids.append((cX, cY, det))

        used_object_ids = set()
        used_detection_indices = set()

        for di, (cX, cY, det) in enumerate(input_centroids):
            object_id = None
            min_dist = float("inf")
            det_label = str(det.get("label", "object"))

            for oid, centroid in self.objects.items():
                if oid in used_object_ids:
                    continue
                dist = np.hypot(float(centroid[0] - cX), float(centroid[1] - cY))
                if dist < min_dist and dist < self.max_distance:
                    track_label = self.labels.get(oid, "object")
                    if track_label != det_label:
                        continue
                    min_dist = dist
                    object_id = oid

            if object_id is not None:
                old_centroid = self.objects[object_id]
                vx = (cX - old_centroid[0]) / dt
                vy = (cY - old_centroid[1]) / dt

                self.objects[object_id] = (cX, cY)
                self.velocities[object_id] = (round(vx, 2), round(vy, 2))
                self.disappeared[object_id] = 0
                used_object_ids.add(object_id)
                used_detection_indices.add(di)

                det["id"] = object_id
                det["centroid"] = [cX, cY]
                det_meta = dict(det.get("meta", {}) or {})
                det_meta["velocity"] = self.velocities[object_id]
                det["meta"] = det_meta
            else:
                self._register(cX, cY, det)
                used_detection_indices.add(di)

        for oid in list(self.disappeared.keys()):
            if oid not in used_object_ids:
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self._deregister(oid)

        return detections

    def _register(self, cX: int, cY: int, det: Dict[str, Any]) -> None:
        object_id = self.next_object_id
        self.next_object_id += 1

        self.objects[object_id] = (cX, cY)
        self.labels[object_id] = str(det.get("label", "object"))
        self.disappeared[object_id] = 0
        self.velocities[object_id] = (0.0, 0.0)

        det["id"] = object_id
        det["centroid"] = [cX, cY]
        det_meta = dict(det.get("meta", {}) or {})
        det_meta["velocity"] = self.velocities[object_id]
        det["meta"] = det_meta

    def _deregister(self, object_id: int) -> None:
        self.objects.pop(object_id, None)
        self.labels.pop(object_id, None)
        self.disappeared.pop(object_id, None)
        self.velocities.pop(object_id, None)

    def get_active_objects(self) -> List[Dict[str, Any]]:
        active = []
        for oid, centroid in self.objects.items():
            active.append({
                "id": oid,
                "centroid": centroid,
                "velocity": self.velocities.get(oid, (0.0, 0.0)),
            })
        return active
