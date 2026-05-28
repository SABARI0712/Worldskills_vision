from __future__ import annotations

from typing import Any, Dict, List

import cv2
import numpy as np

from pyzbar.pyzbar import decode as pyzbar_decode, ZBarSymbol

from ..types import Detection


class BarcodeDetector:
    def __init__(self) -> None:
        self._symbols = [
            ZBarSymbol.EAN13,
            ZBarSymbol.EAN8,
            ZBarSymbol.CODE128,
            ZBarSymbol.CODE39,
            ZBarSymbol.CODE93,
            ZBarSymbol.UPCA,
            ZBarSymbol.UPCE,
            ZBarSymbol.I25,
            ZBarSymbol.CODABAR,
        ]

    @staticmethod
    def _normalize_points(pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=float)

        if pts.ndim == 3 and pts.shape[0] == 1:
            pts = pts[0]

        if pts.ndim == 3 and pts.shape[1] == 1:
            pts = pts.reshape(-1, 2)

        return pts

    def _make_detection(
        self,
        pts: np.ndarray,
        data: str,
        barcode_type: str,
        scale: float = 1.0,
        pad: int = 0,
    ) -> Detection:
        pts = self._normalize_points(pts)

        xs = (pts[:, 0] - pad) / scale
        ys = (pts[:, 1] - pad) / scale

        x1, y1 = int(xs.min()), int(ys.min())
        x2, y2 = int(xs.max()), int(ys.max())

        return Detection(
            label=f"barcode:{data}",
            confidence=1.0 if data else 0.7,
            bbox_xyxy=(x1, y1, x2, y2),
            source="barcode",
            meta={"data": data or "", "type": barcode_type},
        )

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        if frame_bgr is None or frame_bgr.size == 0:
            return []

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        detections: List[Detection] = []
        try:
            decoded = pyzbar_decode(gray, symbols=self._symbols)
            for obj in decoded:
                pts = None
                if hasattr(obj, "polygon") and obj.polygon:
                    pts = np.array([[p.x, p.y] for p in obj.polygon], dtype=np.float32)
                else:
                    pts = np.array(
                        [
                            [obj.rect.left, obj.rect.top],
                            [obj.rect.left + obj.rect.width, obj.rect.top],
                            [obj.rect.left + obj.rect.width, obj.rect.top + obj.rect.height],
                            [obj.rect.left, obj.rect.top + obj.rect.height],
                        ],
                        dtype=np.float32,
                    )

                try:
                    data = obj.data.decode("utf-8")
                except Exception:
                    data = ""

                detections.append(
                    self._make_detection(
                        pts,
                        data,
                        str(obj.type or "UNKNOWN"),
                    )
                )
        except Exception:
            pass

        return detections
