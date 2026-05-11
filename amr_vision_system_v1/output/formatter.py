from __future__ import annotations

from typing import Any, Dict, List, Union

from .csv_formatter import export_csv
from .json_formatter import export_json


def export_output(fmt: str, cfg: Dict[str, Any], payload: Union[Dict[str, Any], List[Dict[str, Any]]]) -> None:
    fmt = (fmt or "json").lower()
    append = bool(cfg.get("append", False))
    if fmt == "json":
        export_json(str(cfg.get("json_path", "results/output.json")), payload, append=append)
        return
    if fmt == "csv":
        if isinstance(payload, dict):
            export_csv(str(cfg.get("csv_path", "results/output.csv")), payload, append=append)
        else:
            raise ValueError("csv export only supports single-frame payloads")
        return
    raise ValueError(f"unknown output format: {fmt}")

