from __future__ import annotations

from typing import Any, Dict, List, Union

from .csv_formatter import export_csv
from .json_formatter import export_json


def export_output(fmt: str, cfg: Dict[str, Any], payload: Union[Dict[str, Any], List[Dict[str, Any]]]) -> None:
    fmt = (fmt or "json").lower()
    append = bool(cfg.get("append", False))

    # Normalise to list internally
    if isinstance(payload, dict):
        payload = [payload]

    if fmt == "json":
        if append:
            # JSON Lines mode: json_formatter expects a bare dict per call.
            # Unwrap single-item list; iterate for multi-frame flush batches.
            for frame in payload:
                export_json(str(cfg.get("json_path", "results/output.json")), frame, append=True)
        else:
            export_json(str(cfg.get("json_path", "results/output.json")), payload, append=False)
        return

    if fmt == "csv":
        export_csv(str(cfg.get("csv_path", "results/output.csv")), payload, append=append)
        return

    raise ValueError(f"unknown output format: {fmt}")

