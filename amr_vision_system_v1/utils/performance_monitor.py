from __future__ import annotations

import time
from typing import Any, Dict, List


class PerformanceMonitor:
    def __init__(self) -> None:
        self.records: List[Dict[str, Any]] = []

    def record(self, stage: str, elapsed_s: float) -> None:
        self.records.append({"stage": stage, "elapsed_s": float(elapsed_s), "timestamp_ms": int(time.time() * 1000)})

    def summary(self) -> Dict[str, Any]:
        totals: Dict[str, Any] = {}
        counts: Dict[str, int] = {}
        for item in self.records:
            stage = str(item.get("stage", "unknown"))
            totals[stage] = totals.get(stage, 0.0) + float(item.get("elapsed_s", 0.0))
            counts[stage] = counts.get(stage, 0) + 1

        return {
            "totals_s": totals,
            "counts": counts,
            "average_s": {stage: totals[stage] / counts[stage] for stage in counts},
        }

    def clear(self) -> None:
        self.records.clear()
