# WorldSkills Vision

Real-time computer vision stack for AMR (Autonomous Mobile Robot) perception — built for WorldSkills-style board detection, object identification, and tracking.

```text
camera → preprocess → detect → fuse → track → map → export → visualize
```

## Project

| Path | Description |
|------|-------------|
| [`amr_vision_system_v1/`](./amr_vision_system_v1/) | Stable modular perception pipeline (main system) |

See the [AMR Vision System V1 README](./amr_vision_system_v1/README.md) for full setup, config, and pipeline details.

## Quick start

```bash
cd amr_vision_system_v1
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 main.py
```

Stop with **ESC** in the OpenCV window, or `Ctrl+C` in the terminal.

## What it does

- Live camera input (USB, ROS2, or MJPEG stream)
- Hybrid detection: YOLO + classical (ArUco, QR, barcode, OCR, color, contour)
- Duplicate fusion with source priority
- Multi-object centroid tracking with velocity
- Optional board-cell mapping (e.g. `A1`, `D4`)
- CSV / JSON export, live overlay, and optional ROS2 publish

## License

Use as needed for competition / development work.
