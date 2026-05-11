from __future__ import annotations

import csv
from typing import Any, Dict, List

from utils.helpers import ensure_parent_dir


def export_csv(path: str, det_rows: List[Dict[str, Any]]) -> None:
    ensure_parent_dir(path)
    fieldnames = ["label", "confidence", "x1", "y1", "x2", "y2", "source", "cell"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in det_rows:
            bbox = r.get("bbox_xyxy", [None, None, None, None])
            row = {
                "label": r.get("label"),
                "confidence": r.get("confidence"),
                "x1": bbox[0],
                "y1": bbox[1],
                "x2": bbox[2],
                "y2": bbox[3],
                "source": r.get("source"),
                "cell": r.get("cell"),
            }
            w.writerow(row)
