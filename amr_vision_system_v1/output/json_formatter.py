from __future__ import annotations

import json
from typing import Any, Dict, List

from utils.helpers import ensure_parent_dir


def export_json(path: str, payload: Dict[str, Any]) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

