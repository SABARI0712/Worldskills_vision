from __future__ import annotations

from typing import Any, Dict, List


REQUIRED_DET_KEYS = {"label", "confidence", "bbox_xyxy", "source"}


def validate_detection(det: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    missing = REQUIRED_DET_KEYS - set(det.keys())
    if missing:
        errors.append(f"missing keys: {sorted(missing)}")
        return errors

    conf = det.get("confidence")
    if not isinstance(conf, (int, float)) or not (0.0 <= float(conf) <= 1.0):
        errors.append("confidence must be float in [0,1]")

    bbox = det.get("bbox_xyxy")
    if (
        not isinstance(bbox, (list, tuple))
        or len(bbox) != 4
        or not all(isinstance(v, (int, float)) for v in bbox)
    ):
        errors.append("bbox_xyxy must be length-4 numeric list/tuple")

    src = det.get("source")
    if not isinstance(src, str) or not src:
        errors.append("source must be non-empty string")

    label = det.get("label")
    if not isinstance(label, str) or not label:
        errors.append("label must be non-empty string")

    return errors


def validate_detections(dets: List[Dict[str, Any]]) -> List[str]:
    errors: List[str] = []
    for i, d in enumerate(dets):
        for e in validate_detection(d):
            errors.append(f"det[{i}]: {e}")
    return errors
