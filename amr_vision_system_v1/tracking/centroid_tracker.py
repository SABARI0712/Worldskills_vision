from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import math


@dataclass
class Track:
    object_id: int
    label: str
    bbox_xyxy: Tuple[int, int, int, int]
    centroid: Tuple[int, int]
    disappeared: int = 0
    history: List[Dict[str, Any]] = field(default_factory=list)

    @staticmethod
    def compute_centroid(bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        x1, y1, x2, y2 = bbox
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))

    def update(self, bbox: Tuple[int, int, int, int], label: str, meta: Optional[Dict[str, Any]] = None) -> None:
        self.bbox_xyxy = bbox
        self.centroid = self.compute_centroid(bbox)
        self.label = label
        self.disappeared = 0
        self.history.append({"bbox_xyxy": bbox, "label": label, "meta": dict(meta or {})})


class CentroidTracker:
    def __init__(
        self,
        max_disappeared: int = 10,
        max_distance: int = 80,
        label_penalty: float = 100.0,
    ) -> None:
        self.max_disappeared = int(max_disappeared)
        self.max_distance = float(max_distance)
        self.label_penalty = float(label_penalty)
        self.next_object_id = 1
        self.tracks: Dict[int, Track] = {}

    def _register(self, detection: Dict[str, Any]) -> None:
        bbox = tuple(map(int, detection.get("bbox_xyxy", [0, 0, 0, 0])))
        label = str(detection.get("label", "object"))
        track_id = self.next_object_id
        self.next_object_id += 1

        track = Track(
            object_id=track_id,
            label=label,
            bbox_xyxy=bbox,
            centroid=Track.compute_centroid(bbox),
        )
        track.history.append({"bbox_xyxy": bbox, "label": label, "meta": dict(detection.get("meta", {}) or {})})
        self.tracks[track_id] = track
        detection["id"] = track_id

    def _deregister(self, object_id: int) -> None:
        if object_id in self.tracks:
            del self.tracks[object_id]

    def _distance(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return math.hypot(float(a[0] - b[0]), float(a[1] - b[1]))

    def _match(self, track_centroids: List[Tuple[int, int]], detections: List[Dict[str, Any]]) -> List[Tuple[int, int]]:
        pairs: List[Tuple[float, int, int]] = []
        for ti, track_centroid in enumerate(track_centroids):
            for di, det in enumerate(detections):
                det_bbox = tuple(map(int, det.get("bbox_xyxy", [0, 0, 0, 0])))
                det_centroid = Track.compute_centroid(det_bbox)
                distance = self._distance(track_centroid, det_centroid)
                if distance > self.max_distance:
                    continue
                track_id = list(self.tracks.keys())[ti]
                track_label = self.tracks[track_id].label
                det_label = str(det.get("label", "object"))
                if track_label != det_label:
                    distance += self.label_penalty
                pairs.append((distance, ti, di))

        pairs.sort(key=lambda x: x[0])
        assigned_tracks = set()
        assigned_detections = set()
        matches: List[Tuple[int, int]] = []

        for distance, ti, di in pairs:
            if ti in assigned_tracks or di in assigned_detections:
                continue
            if distance > self.max_distance + self.label_penalty:
                continue
            assigned_tracks.add(ti)
            assigned_detections.add(di)
            matches.append((ti, di))

        return matches

    def _mark_disappeared(self, unmatched_track_ids: List[int]) -> None:
        for object_id in unmatched_track_ids:
            track = self.tracks.get(object_id)
            if track is None:
                continue
            track.disappeared += 1
            if track.disappeared > self.max_disappeared:
                self._deregister(object_id)

    def update(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if detections is None:
            detections = []

        if len(detections) == 0:
            self._mark_disappeared(list(self.tracks.keys()))
            return []

        if not self.tracks:
            for detection in detections:
                self._register(detection)
            return detections

        track_ids = list(self.tracks.keys())
        track_centroids = [self.tracks[track_id].centroid for track_id in track_ids]
        matches = self._match(track_centroids, detections)

        matched_tracks = {track_ids[ti] for ti, _ in matches}
        matched_detections = {di for _, di in matches}

        for ti, di in matches:
            object_id = track_ids[ti]
            detection = detections[di]
            bbox = tuple(map(int, detection.get("bbox_xyxy", [0, 0, 0, 0])))
            label = str(detection.get("label", "object"))
            self.tracks[object_id].update(bbox, label, detection.get("meta", {}))
            detection["id"] = object_id

        unmatched_track_ids = [track_id for track_id in track_ids if track_id not in matched_tracks]
        self._mark_disappeared(unmatched_track_ids)

        for di, detection in enumerate(detections):
            if di in matched_detections:
                continue
            self._register(detection)

        return detections
