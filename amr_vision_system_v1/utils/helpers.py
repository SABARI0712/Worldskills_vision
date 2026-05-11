from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Tuple


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


@dataclass
class Timer:
    start_s: float = 0.0

    def __enter__(self) -> "Timer":
        self.start_s = time.time()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        pass

    @property
    def elapsed_s(self) -> float:
        return time.time() - self.start_s


def clamp_box_xyxy(box: Tuple[int, int, int, int], w: int, h: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w - 1))
    y2 = max(0, min(int(y2), h - 1))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def now_ms() -> int:
    return int(time.time() * 1000)


def deep_get(d: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur
