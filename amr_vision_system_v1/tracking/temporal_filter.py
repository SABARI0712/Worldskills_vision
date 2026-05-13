from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


class TemporalFilter:
    def __init__(self, alpha: float = 0.7) -> None:
        self.alpha = float(alpha)
        self.states: Dict[int, Dict[str, Any]] = {}

    @staticmethod
    def _blend(value: float, previous: float, alpha: float) -> float:
        return alpha * value + (1.0 - alpha) * previous

    @staticmethod
    def _smooth_bbox(
        previous: Tuple[int, int, int, int],
        current: Tuple[int, int, int, int],
        alpha: float,
    ) -> Tuple[int, int, int, int]:
        return (
            int(TemporalFilter._blend(current[0], previous[0], alpha)),
            int(TemporalFilter._blend(current[1], previous[1], alpha)),
            int(TemporalFilter._blend(current[2], previous[2], alpha)),
            int(TemporalFilter._blend(current[3], previous[3], alpha)),
        )

    @staticmethod
    def _blend_angle(current: float, previous: float, alpha: float) -> float:
        """Blend angles on a 180-degree circular domain to avoid flip artifacts."""
        delta = ((current - previous + 90.0) % 180.0) - 90.0
        blended = previous + alpha * delta
        blended = blended % 180.0
        if blended < 0.0:
            blended += 180.0
        return blended

    def update(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        updated: List[Dict[str, Any]] = []
        active_ids: List[int] = []

        for detection in detections:
            object_id = detection.get("id")
            bbox = tuple(map(int, detection.get("bbox_xyxy", [0, 0, 0, 0])))
            confidence = float(detection.get("confidence", 0.0))

            if object_id is None:
                updated.append(detection)
                continue

            state = self.states.get(object_id)
            if state is None:
                self.states[object_id] = {"bbox": bbox, "confidence": confidence}
                # Initialize angle if present
                angle = detection.get("angle")
                if angle is not None:
                    self.states[object_id]["angle"] = float(angle)
            else:
                smoothed_bbox = self._smooth_bbox(state["bbox"], bbox, self.alpha)
                smoothed_confidence = self._blend(confidence, float(state["confidence"]), self.alpha)
                detection["bbox_xyxy"] = [int(x) for x in smoothed_bbox]
                detection["confidence"] = float(smoothed_confidence)

                # Smooth angle if present, preserving previous orientation when current angle is missing
                angle = detection.get("angle")
                state_angle = state.get("angle")
                if angle is not None:
                    if state_angle is not None:
                        detection["angle"] = self._blend_angle(float(angle), float(state_angle), self.alpha)
                    else:
                        detection["angle"] = float(angle)
                    state_angle = detection["angle"]

                self.states[object_id] = {
                    "bbox": smoothed_bbox,
                    "confidence": smoothed_confidence,
                    "angle": state_angle,
                }

            active_ids.append(object_id)
            updated.append(detection)

        self.states = {oid: state for oid, state in self.states.items() if oid in active_ids}
        return updated
