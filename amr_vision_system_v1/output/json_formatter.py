from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Union

from utils.helpers import ensure_parent_dir


def export_json(path: str, payload: Union[Dict[str, Any], List[Dict[str, Any]]], *, append: bool = False) -> None:
    ensure_parent_dir(path)
    if append:
        if not isinstance(payload, dict):
            raise ValueError("append mode only supports single-frame output payloads")
            
        # If the file does not exist or is empty, write a fresh JSON array with the first element
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            with open(path, "w", encoding="utf-8") as f:
                f.write("[\n  ")
                json.dump(payload, f, ensure_ascii=False)
                f.write("\n]")
            return

        # Open in read-write binary mode to safely seek, truncate, and append to the JSON array
        with open(path, "r+b") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            
            # Read backwards from the end to find the closing bracket ']'
            chunk_size = min(size, 1024)
            f.seek(-chunk_size, os.SEEK_END)
            chunk = f.read(chunk_size)
            
            r_index = chunk.rfind(b']')
            if r_index != -1:
                truncate_pos = size - chunk_size + r_index
                
                # Check if this is an empty array (e.g. "[]" or "[\n]")
                f.seek(0)
                content_before = f.read(truncate_pos)
                is_empty = content_before.strip(b'\r\n\t ') == b'['
                
                f.seek(truncate_pos)
                f.truncate()
                
                prefix = "\n  " if is_empty else ",\n  "
                new_item_str = prefix + json.dumps(payload, ensure_ascii=False) + "\n]"
                f.write(new_item_str.encode("utf-8"))
            else:
                # Fallback if no closing bracket is found
                f.seek(0)
                try:
                    content = f.read().decode("utf-8").strip()
                    data = json.loads(content) if content else []
                    if isinstance(data, list):
                        data.append(payload)
                    else:
                        data = [payload]
                except Exception:
                    data = [payload]
                
                f.seek(0)
                f.truncate()
                f.write(json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8"))
        return

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
