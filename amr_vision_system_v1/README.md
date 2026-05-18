## AMR Vision System V1 (stable modular pipeline)

This folder is a **new, separate** implementation of your Version 1 perception stack:

```text
camera → preprocess → detect → export → visualize
```

It is designed to be **stable and modular** and includes tracking, pose estimation, fusion, and visualization components.

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
- `config/loader.py` loads YAML into a dictionary used system-wide.

### 2) Camera
- `camera/camera_handler.py` selects and manages:
  - `USBCamera` (`camera/usb_camera.py`) with Linux V4L2 backend, auto-reconnect, and custom Orbbec Gemini E parameter optimizations (autofocus disable, manual focus, exposure/gain tuning).
  - `ROSCamera` (`camera/ros_camera.py`) for ROS2 topic integration with `cv_bridge`.

### 3) Preprocess
- `preprocessing/image_preprocessor.py` applies optional/configurable steps:
  - Spatial resizing (maintaining aspect ratio or forced dimensions).
  - CLAHE (Contrast Limited Adaptive Histogram Equalization) in LAB colorspace.
  - Non-local means denoising to clean image noise.

### 4) Detect
- `detection/hybrid_detector.py` orchestrates object detection:
  - `mode: yolo` runs `detection/ml/yolo_detector.py` with custom weights.
  - `mode: classical` runs QR code, ArUco, OCR, and color detectors concurrently.
  - `mode: hybrid` merges semantic deep learning and classical detections, leveraging semantic protection regions to prevent overlap.

### 5) Fusion & Resolution
- `fusion/detection_fuser.py` and `DuplicateResolver` (`fusion/duplicate_resolver.py`):
  - Resolves overlapping bounding boxes via IoU thresholds.
  - Prioritizes highly reliable sensors: `aruco` > `qr` > `ocr` > `yolo` > `color` > `contour`.
  - Intelligently filters and preserves specific color labels over generic background components.

### 6) Tracking & Filtering
- `tracking/centroid_tracker.py` (`CentroidTracker`):
  - Associates bounding boxes across frames using centroid Euclidean distances.
  - Tracks ID persistence with custom disappearing limits.
  - Computes real-time 2D pixel velocities using frame timestamps.
- `tracking/temporal_filter.py` (`TemporalFilter`):
  - Exponentially smooths bounding boxes and confidence scores over time.
  - Blend orientations (angles) on a 180-degree circular domain to prevent flip artifacts.
  - Integrates with tracker liveness to persist states smoothly through transient frames.

### 7) Postprocess & Pose Estimation
- `postprocessing/post_processor.py` manages coordinate clamping and confidence sorting.
- `postprocessing/pose_estimator.py` (`PoseEstimator`):
  - Uses contour moments to compute 2D/3D physical orientations and centroids.

### 8) Mapping & Spatial Analytics
- `postprocessing/board_mapper.py` (`BoardMapper`):
  - Maps 2D pixel coordinates to discrete board grid coordinates (e.g. A1, D5) based on a configurable matrix.
- `perception/occupancy_grid.py` (`OccupancyGrid`):
  - Translates active tracker states into occupancy maps.
- `perception/counter.py` (`ObjectCounter`):
  - Compiles structured frame-by-frame summaries of class, color, and cell distributions.
- `perception/world_model.py` (`WorldModel`) & `SceneMemory` (`perception/scene_memory.py`):
  - WorldModel maintains global tracking states, history buffers, and latest known positions.
  - SceneMemory maintains a historical window of perception frames for retrospective reasoning.

### 9) Export
- `output/formatter.py` routing:
  - `output/json_formatter.py`: exports to a standard JSON array iteratively with `O(1)` back-seek and truncate logic for memory safety and format correctness.
  - `output/csv_formatter.py`: exports frame statistics incrementally in CSV format.

### 10) Visualize
- `output/visualizer.py`:
  - Renders colored bounding boxes, centroids, tracking IDs, and custom pose orientation vectors.
  - Renders board overlays, cell grids, and text summaries.

---

## Extension points (keep V1 stable)
When you extend later, do it by adding modules, not by mixing responsibilities:
- Add new classical detectors in `detection/classical/`
- Add additional postprocessing rules in `postprocessing/`
- Add new exporters in `output/` and route via `output/formatter.py`

