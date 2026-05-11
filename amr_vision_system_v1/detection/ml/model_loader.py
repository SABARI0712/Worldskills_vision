from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=4)
def load_yolo_model(weights_path: str):
    from ultralytics import YOLO

    return YOLO(weights_path)

