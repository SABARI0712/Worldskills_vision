from __future__ import annotations

import json
from typing import Any, Dict, List, Union

from utils.helpers import ensure_parent_dir


def export_json(path: str, payload: Union[Dict[str, Any], List[Dict[str, Any]]], *, append: bool = False) -> None:
    ensure_parent_dir(path)
    if append:
        # JSON Lines: one JSON object per line (frame history).
        if not isinstance(payload, dict):
            raise ValueError("append mode only supports single-frame output payloads")
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False))
            f.write("\n")
        return

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

