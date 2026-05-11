from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, List, Union

from utils.helpers import ensure_parent_dir


def export_csv(path: str, payload: Union[Dict[str, Any], List[Dict[str, Any]]], *, append: bool = False) -> None:
    ensure_parent_dir(path)

    # Handle both single-frame dict and multi-frame list
    if isinstance(payload, dict):
        frames = [payload]
    elif isinstance(payload, list):
        frames = payload
    else:
        raise ValueError("payload must be dict or list of dicts")

    fieldnames = [
        "timestamp_ms",
        "image_width",
        "image_height",
        "label",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
        "source",
        "cell",
        "meta_json",
    ]

    file_exists = os.path.exists(path)
    write_header = (not file_exists) or os.path.getsize(path) == 0
    mode = "a" if append else "w"

    with open(path, mode, newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()

        for frame_payload in frames:
            ts = frame_payload.get("timestamp_ms")
            img = frame_payload.get("image") or {}
            iw = img.get("width")
            ih = img.get("height")
            det_rows = frame_payload.get("detections", []) or []

            for r in det_rows:
                bbox = r.get("bbox_xyxy", [None, None, None, None])
                row = {
                    "timestamp_ms": ts,
                    "image_width": iw,
                    "image_height": ih,
                    "label": r.get("label"),
                    "confidence": r.get("confidence"),
                    "x1": bbox[0],
                    "y1": bbox[1],
                    "x2": bbox[2],
                    "y2": bbox[3],
                    "source": r.get("source"),
                    "cell": r.get("cell"),
                    "meta_json": json.dumps(r.get("meta", {}) or {}, ensure_ascii=False),
                }
                w.writerow(row)
