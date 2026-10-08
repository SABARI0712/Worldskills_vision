import os
import json
import tempfile
from output.json_formatter import export_json

def test_export_json_normal_write():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "out.json")
        payload = {"timestamp_ms": 1000, "detections": [{"id": 1, "label": "test"}]}
        
        # Write fresh
        export_json(path, payload, append=False)
        
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        assert data == payload

def test_export_json_append_increments():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "out.json")
        
        # Write first frame
        export_json(path, {"frame": 1, "val": "A"}, append=True)
        # Write second frame
        export_json(path, {"frame": 2, "val": "B"}, append=True)
        # Write third frame
        export_json(path, {"frame": 3, "val": "C"}, append=True)
        
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        assert isinstance(data, list)
        assert len(data) == 3
        assert data[0]["frame"] == 1
        assert data[1]["frame"] == 2
        assert data[2]["frame"] == 3
        assert data[2]["val"] == "C"

def test_export_json_append_empty_preexisting():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "out.json")
        
        # Scenario A: empty file
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        export_json(path, {"frame": 1}, append=True)
        
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data == [{"frame": 1}]

        # Scenario B: "[]"
        with open(path, "w", encoding="utf-8") as f:
            f.write("[]")
        export_json(path, {"frame": 2}, append=True)
        export_json(path, {"frame": 3}, append=True)
        
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data == [{"frame": 2}, {"frame": 3}]

        # Scenario C: "[\n]"
        with open(path, "w", encoding="utf-8") as f:
            f.write("[\n]")
        export_json(path, {"frame": 4}, append=True)
        
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data == [{"frame": 4}]
