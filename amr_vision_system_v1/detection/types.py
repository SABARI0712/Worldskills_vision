from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass
class Detection:
    label: str
    confidence: float
    bbox_xyxy: Tuple[int, int, int, int]
    source: str
    meta: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "confidence": float(self.confidence),
            "bbox_xyxy": [int(self.bbox_xyxy[0]), int(self.bbox_xyxy[1]), int(self.bbox_xyxy[2]), int(self.bbox_xyxy[3])],
            "source": self.source,
            "meta": dict(self.meta or {}),
        }


def detections_to_dicts(dets: List[Detection]) -> List[Dict[str, Any]]:
    return [d.to_dict() for d in dets]

