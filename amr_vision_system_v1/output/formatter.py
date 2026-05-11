from __future__ import annotations

from typing import Any, Dict, List

from .csv_formatter import export_csv
from .json_formatter import export_json


def export_output(fmt: str, cfg: Dict[str, Any], payload: Dict[str, Any]) -> None:
    fmt = (fmt or "json").lower()
    if fmt == "json":
        export_json(str(cfg.get("json_path", "results/output.json")), payload)
        return
    if fmt == "csv":
        export_csv(str(cfg.get("csv_path", "results/output.csv")), payload.get("detections", []))
        return
    raise ValueError(f"unknown output format: {fmt}")

