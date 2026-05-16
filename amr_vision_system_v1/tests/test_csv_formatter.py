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
