from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from ..types import Detection
from .model_loader import load_yolo_model


class YOLODetector:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.weights_path = str(cfg.get("weights_path", "models/best.pt"))
        self.imgsz = int(cfg.get("imgsz", 640))
        self.device = cfg.get("device", None)
        self.iou = float(cfg.get("iou", 0.45))
        self.max_det = int(cfg.get("max_det", 50))

        self._model = None

    def warmup(self) -> None:
        self._model = load_yolo_model(self.weights_path)

    def detect(self, frame_bgr: np.ndarray, conf: float) -> List[Detection]:
        if self._model is None:
            self.warmup()

        model = self._model
        results = model.predict(
            source=frame_bgr,
            conf=float(conf),
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            max_det=self.max_det,
            verbose=False,
        )

        dets: List[Detection] = []
        h, w = frame_bgr.shape[:2]
        names = getattr(model, "names", {}) or {}

        for r in results:
            boxes = getattr(r, "boxes", None)
            if boxes is None:
                continue
            for b in boxes:
                try:
                    c = float(b.conf[0])
                    cls = int(b.cls[0])
                    label = names.get(cls, str(cls))
                    x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                except Exception:
                    continue

                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(0, min(x2, w - 1))
                y2 = max(0, min(y2, h - 1))

                bbox_width = x2 - x1
                bbox_height = y2 - y1
                bbox_area = bbox_width * bbox_height

                dets.append(
                    Detection(
                        label=str(label),
                        confidence=float(c),
                        bbox_xyxy=(x1, y1, x2, y2),
                        source="yolo",
                        meta={
                            "class_id": cls,
                            "area": bbox_area,
                            "width": bbox_width,
                            "height": bbox_height,
                        },
                    )
                )

        return dets
