# AMR Vision System V1

Stable, modular real-time perception pipeline for an Autonomous Mobile Robot (AMR).

```text
camera → preprocess → perspective → detect → color classify
       → postprocess → fuse → track → pose → temporal filter
       → board map → world model → export / visualize / ROS
```

---

## Features

- **Camera sources:** USB (V4L2), ROS2 topic, MJPEG HTTP stream
- **Detection modes:** `yolo` | `classical` | `hybrid`
- **Classical detectors:** ArUco, QR, barcode, OCR, HSV color, contour
- **ML detector:** YOLO (Ultralytics) with custom weights
- **Fusion:** IoU-based duplicate resolution with source priority  
  `aruco > qr > ocr > yolo > color > contour`
- **Tracking:** Centroid tracker + temporal smoothing + velocity
- **Spatial:** Optional perspective warp and grid/board cell mapping
- **Outputs:** CSV / JSON, annotated frames, terminal lines, ROS2 topics

---

## Folder layout

```text
amr_vision_system_v1/
  main.py                 # Entry point / frame loop
  requirements.txt
  config/
    config.yaml           # All runtime settings
    loader.py
  camera/                 # USB, ROS, MJPEG (+ threaded reader)
  preprocessing/          # Resize, CLAHE, denoise
  detection/
    hybrid_detector.py    # Orchestrator + scheduling
    types.py
    classical/            # ArUco, QR, barcode, OCR, color, contour
    ml/                   # YOLO
  fusion/                 # Duplicate resolver
  tracking/               # Centroid tracker, temporal filter
  postprocessing/         # Post-filter, pose, board map, perspective
  perception/             # Color classifier, world model, occupancy, counter
  output/                 # CSV/JSON, visualizer, ROS publisher
  utils/                  # Logger, helpers, validators, perf monitor
  models/                 # Place best.pt here
  results/                # output.csv / output.json / annotated.jpg
  logs/                   # runtime_*.log
  tests/
```

---

## Install

```bash
cd amr_vision_system_v1
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

| Use case | Extra needs |
|----------|-------------|
| USB camera only | OpenCV, NumPy, PyYAML (base `requirements.txt`) |
| YOLO | `ultralytics` + weights at `models/best.pt` |
| OCR | `pytesseract` + system Tesseract |
| ROS camera / publish | ROS2 with `rclpy` and `cv_bridge` |

---

## Configure

Edit [`config/config.yaml`](./config/config.yaml).

### Important keys

| Key | Options / meaning |
|-----|-------------------|
| `camera.source` | `usb` \| `ros` \| `mjpeg` |
| `preprocess.enabled` | Resize / CLAHE / denoise |
| `perspective.enabled` | Top-down warp |
| `detection.mode` | `yolo` \| `classical` \| `hybrid` |
| `detection.confidence` | Global confidence floor |
| `detection.yolo.enabled` | Toggle YOLO |
| `detection.classical.*` | Toggle each classical detector |
| `mapping.enabled` | Pixel → board cell (e.g. `A8`) |
| `output.format` | `json` \| `csv` |
| `output.append` | Per-frame append (recommended) |
| `visualization.show_window` | Live OpenCV window |
| `terminal_output` | Which fields to print each frame |

### YOLO weights

```yaml
detection:
  yolo:
    enabled: true
    weights_path: models/best.pt
```

Place weights at `models/best.pt` (absolute paths also work).

### Optional config path override

```bash
export AMR_VISION_CONFIG=/path/to/custom.yaml
python3 main.py
```

---

## Run

```bash
cd amr_vision_system_v1
python3 main.py
```

Stop with **ESC** in the visualization window, or `Ctrl+C` in the terminal.

---

## Pipeline overview

1. **Config** — `config/loader.py` loads YAML for the whole system.
2. **Camera** — `camera/camera_handler.py` creates USB / ROS / MJPEG (optional threaded reader).
3. **Preprocess** — optional resize, CLAHE (LAB), denoise.
4. **Perspective** — optional 4-point warp to a top-down view.
5. **Detect** — `HybridDetector` runs enabled detectors on a schedule:
   - ArUco every frame
   - YOLO / QR / barcode / OCR / color / contour on intervals
   - Adaptive backoff when detection is slow (> ~80 ms)
   - ROI crops from YOLO/ArUco for QR, barcode, OCR
   - Cache carry-over on skipped frames for smooth tracking
6. **Color classify** — sample color inside boxes for YOLO / QR / ArUco / contour.
7. **Postprocess** — confidence filter + bbox clamp (optional).
8. **Fuse** — resolve overlaps; prefer higher-priority sources.
9. **Track** — persistent IDs, centroids, pixel velocity.
10. **Pose** — orientation from contour moments (optional).
11. **Temporal filter** — smooth boxes / confidence / angle.
12. **Map** — assign board cells from centroid (optional).
13. **World model** — live object state, occupancy, counts, scene memory.
14. **Export / view** — CSV or JSON, overlay window, terminal lines, ROS topics.

---

## Outputs

| Artifact | Location |
|----------|----------|
| JSON | `results/output.json` (when `output.format: json`) |
| CSV | `results/output.csv` (when `output.format: csv`) |
| Annotated frame | `results/annotated.jpg` (if `save_image: true`) |
| Logs | `logs/runtime_*.log` |

### JSON frame schema (example)

```json
{
  "timestamp_ms": 1234567890,
  "frame": 42,
  "image": { "width": 640, "height": 480 },
  "detections": [
    {
      "label": "box",
      "confidence": 0.82,
      "bbox_xyxy": [10, 20, 300, 220],
      "source": "yolo",
      "meta": { "class_id": 0, "velocity": [0.0, 0.0] },
      "color": "red",
      "id": 1,
      "centroid": [155, 120],
      "cell": "D4"
    }
  ],
  "detection_count": 1,
  "occupancy": {},
  "world_counts": { "box": 1 }
}
```

Notes:
- `cell` appears only when `mapping.enabled: true`.
- Classical labels look like `qr:<data>`, `aruco:<id>`, `color:<name>`.

---

## Tests

```bash
cd amr_vision_system_v1
pytest
```

---

## Extending (keep V1 modular)

- New classical detectors → `detection/classical/`
- New post rules → `postprocessing/`
- New exporters → `output/` and wire through `output/formatter.py`
- Prefer adding modules over mixing responsibilities into `main.py`
