from __future__ import annotations

from typing import Any, Dict, List

from detection.types import Detection
from utils.helpers import clamp_box_xyxy


class PostProcessor:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.min_confidence = float(cfg.get("min_confidence", 0.5))

    def process(self, dets: List[Detection], frame_shape_hw) -> List[Detection]:
        h, w = frame_shape_hw
        out: List[Detection] = []
        for d in dets:
            if float(d.confidence) < self.min_confidence:
                continue
            x1, y1, x2, y2 = clamp_box_xyxy(d.bbox_xyxy, w=w, h=h)
            out.append(
                Detection(
                    label=d.label,
                    confidence=float(d.confidence),
                    bbox_xyxy=(x1, y1, x2, y2),
                    source=d.source,
                    meta=dict(d.meta or {}),
                )
            )
        return out

