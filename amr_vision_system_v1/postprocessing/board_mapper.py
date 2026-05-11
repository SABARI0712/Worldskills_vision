from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass
class GridSpec:
    rows: int
    cols: int
    origin: str = "top_left"
    col_labels: str = "ABCDEFGH"
    row_labels: str = "87654321"


class BoardMapper:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        g = (cfg.get("grid") or {}) if cfg else {}
        labels = (g.get("labels") or {}) if g else {}
        self.spec = GridSpec(
            rows=int(g.get("rows", 8)),
            cols=int(g.get("cols", 8)),
            origin=str(g.get("origin", "top_left")),
            col_labels=str(labels.get("cols", "ABCDEFGH")),
            row_labels=str(labels.get("rows", "87654321")),
        )

    def pixel_to_cell(self, x: int, y: int, frame_w: int, frame_h: int) -> Optional[str]:
        if frame_w <= 0 or frame_h <= 0:
            return None
        col = int((x / frame_w) * self.spec.cols)
        row = int((y / frame_h) * self.spec.rows)
        col = max(0, min(col, self.spec.cols - 1))
        row = max(0, min(row, self.spec.rows - 1))

        if col >= len(self.spec.col_labels) or row >= len(self.spec.row_labels):
            return None
        return f"{self.spec.col_labels[col]}{self.spec.row_labels[row]}"

