## AMR Vision System V1 (stable modular pipeline)

This folder is a **new, separate** implementation of your Version 1 perception stack:

```text
camera → preprocess → detect → export → visualize
```

It is designed to be **stable and modular**. No tracking, no ROS publishing, no pose estimation, no benchmarking, no fail-safe, no multi-camera.

---

## Folder layout

```text
amr_vision_system_v1/
  main.py
  requirements.txt
  config/
    config.yaml
    loader.py
  camera/
    camera_handler.py
    usb_camera.py
    ros_camera.py
    utils.py
  preprocessing/
    image_preprocessor.py
  detection/
    hybrid_detector.py
    types.py
    classical/
      qr_detector.py
      color_detector.py
      contour_detector.py
    ml/
      model_loader.py
      yolo_detector.py
  postprocessing/
    post_processor.py
    board_mapper.py
  output/
    formatter.py
    json_formatter.py
    csv_formatter.py
    visualizer.py
  utils/
    logger.py
    helpers.py
    validators.py
  models/
    best.pt   (put your weights here)
  results/
    output.json / output.csv / annotated.jpg (auto-written)
  logs/
    runtime_*.log
```

---

## Install

From the parent directory:

```bash
cd amr_vision_system_v1
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Notes:
- **USB-only** needs just OpenCV/Numpy/YAML.
- **YOLO** needs `ultralytics` and a weights file (default `models/best.pt`).
- **ROS camera** needs a working ROS2 install with `rclpy` and `cv_bridge` in your environment.

---

## Configure

Edit `config/config.yaml`. Key settings:

- `camera.source`: `usb` or `ros`
- `preprocess.*`: resize/CLAHE/denoise toggles
- `detection.mode`: `yolo` | `classical` | `hybrid`
- `detection.confidence`: global confidence threshold for YOLO
- `output.format`: `json` or `csv`
- `visualization.show_window`: open a live OpenCV window

### YOLO weights

Set:

```yaml
detection:
  yolo:
    weights_path: models/best.pt
```

Place your file at `amr_vision_system_v1/models/best.pt`.

---

## Run

```bash
cd amr_vision_system_v1
python3 main.py
```

Stop:
- Press **ESC** in the visualization window, or Ctrl+C in terminal.

---

## Outputs (Version 1)

Every frame, the pipeline produces:

- `results/output.json` (if `output.format: json`)
- `results/output.csv` (if `output.format: csv`)
- `results/annotated.jpg` (latest annotated frame)
- `logs/runtime_*.log`

### Output schema (JSON)

```json
{
  "timestamp_ms": 1234567890,
  "image": { "width": 640, "height": 480 },
  "detections": [
    {
      "label": "box",
      "confidence": 0.82,
      "bbox_xyxy": [10, 20, 300, 220],
      "source": "yolo",
      "meta": { "class_id": 0 },
      "cell": "D4"
    }
  ]
}
```

Notes:
- `cell` is only added when `mapping.enabled: true`.
- Classical detectors use labels like `qr:<data>` or `color:<range_name>`.

---

## How the pipeline is wired (Version 1)

### 1) Config

- `config/loader.py` loads YAML into a dict used everywhere.

### 2) Camera

- `camera/camera_handler.py` selects:
  - `USBCamera` (`camera/usb_camera.py`) for webcam capture + reconnect
  - `ROSCamera` (`camera/ros_camera.py`) for latest-frame retrieval from ROS2 topic

### 3) Preprocess

- `preprocessing/image_preprocessor.py` runs (configurable):
  - resize
  - CLAHE (in LAB space)
  - optional denoise

### 4) Detect

- `detection/hybrid_detector.py` is the entry point:
  - `mode: yolo` uses `detection/ml/yolo_detector.py`
  - `mode: classical` uses `detection/classical/*`
  - `mode: hybrid` runs both and concatenates detections

### 5) Postprocess

- `postprocessing/post_processor.py`:
  - confidence filtering
  - bbox clamping

### 6) Mapping (optional)

- `postprocessing/board_mapper.py`:
  - maps bbox center pixel \(\rightarrow\) grid cell label (basic)

### 7) Export

- `output/formatter.py` chooses:
  - `output/json_formatter.py`
  - `output/csv_formatter.py`

### 8) Visualize

- `output/visualizer.py`:
  - draws bbox + label + confidence
  - optional grid overlay
  - writes `results/annotated.jpg`

---

## Extension points (keep V1 stable)

When you extend later, do it by adding modules, not by mixing responsibilities:

- Add new classical detectors in `detection/classical/`
- Add additional postprocessing rules in `postprocessing/`
- Add new exporters in `output/` and route via `output/formatter.py`

