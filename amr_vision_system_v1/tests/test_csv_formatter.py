import os
import tempfile

from output.csv_formatter import export_csv


def test_export_csv_adds_header_if_missing():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "out.csv")
        # create a file that already has row data but no header
        with open(path, "w", encoding="utf-8") as f:
            f.write("1,1000,640,480,servo_power_block,0.9,10,10,100,100,yolo,,{}\n")

        export_csv(path, {"timestamp_ms": 1001, "image": {"width": 640, "height": 480}, "detections": []}, append=True)

        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        assert lines[0].startswith("object_id,timestamp_ms,image_width,image_height,label,confidence")
        assert len(lines) == 2


def test_export_csv_overwrite_writes_header():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "out.csv")
        # create a file that already has a header and row data
        with open(path, "w", encoding="utf-8") as f:
            f.write("object_id,timestamp_ms,image_width,image_height,label,confidence,x1,y1,x2,y2,source,cell,meta_json\n")
            f.write("1,1000,640,480,servo_power_block,0.9,10,10,100,100,yolo,,{}\n")

        # Overwrite with append=False
        export_csv(path, {"timestamp_ms": 1001, "image": {"width": 640, "height": 480}, "detections": []}, append=False)

        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        assert len(lines) == 1
        assert lines[0].startswith("object_id,timestamp_ms,image_width,image_height,label,confidence")

