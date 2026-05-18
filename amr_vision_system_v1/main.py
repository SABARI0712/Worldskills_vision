from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, List

import cv2
import numpy as np

from config.loader import load_config
from perception.counter import ObjectCounter
from perception.occupancy_grid import OccupancyGrid
from perception.world_model import WorldModel
from perception.scene_memory import SceneMemory
from camera.camera_handler import create_camera
from preprocessing.image_preprocessor import ImagePreprocessor
from detection.hybrid_detector import HybridDetector
from detection.types import detections_to_dicts
from postprocessing.post_processor import PostProcessor
from postprocessing.board_mapper import BoardMapper
from postprocessing.perspective_transform import PerspectiveTransform
from postprocessing.pose_estimator import PoseEstimator
from output.formatter import export_output
from output.json_formatter import export_json
from output.csv_formatter import export_csv
from output.visualizer import Visualizer
from output.runtime_monitor import RuntimeMonitor
from fusion.detection_fuser import DetectionFuser
from tracking.centroid_tracker import CentroidTracker
from tracking.temporal_filter import TemporalFilter
from perception.color_classifier import ColorClassifier
from utils.helpers import now_ms
from utils.logger import setup_logger
from utils.validators import validate_detections
from utils.performance_monitor import PerformanceMonitor


def _resolve_path(base_dir: str, maybe_rel: str) -> str:
    if os.path.isabs(maybe_rel):
        return maybe_rel
    return os.path.normpath(os.path.join(base_dir, maybe_rel))


def _side_by_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if left.shape[0] != right.shape[0]:
        height = min(left.shape[0], right.shape[0])
        left = cv2.resize(left, (int(left.shape[1] * height / left.shape[0]), height), interpolation=cv2.INTER_AREA)
        right = cv2.resize(right, (int(right.shape[1] * height / right.shape[0]), height), interpolation=cv2.INTER_AREA)
    return cv2.hconcat([left, right])


def main() -> int:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.environ.get("AMR_VISION_CONFIG", os.path.join(base_dir, "config", "config.yaml"))
    cfg = load_config(cfg_path)

    log_dir = _resolve_path(base_dir, str((cfg.get("app") or {}).get("log_dir", "logs")))
    results_dir = _resolve_path(base_dir, str((cfg.get("app") or {}).get("results_dir", "results")))
    os.makedirs(results_dir, exist_ok=True)

    logger = setup_logger(log_dir=log_dir)
    logger.info(f"config: {cfg_path}")

    camera = create_camera(cfg.get("camera") or {})
    pre_cfg = cfg.get("preprocess") or {}
    pre = ImagePreprocessor(pre_cfg) if bool(pre_cfg.get("enabled", True)) else None

    perspective_cfg = cfg.get("perspective") or {}
    perspective = PerspectiveTransform(perspective_cfg)

    detector = HybridDetector(cfg.get("detection") or {})
    detector.warmup()
    logger.info("detector warmed up")

    color_cfg = cfg.get("detection", {}).get("classical", {}).get("color", {})
    classifier = ColorClassifier(color_cfg)

    post_cfg = cfg.get("postprocess") or {}
    post = PostProcessor(post_cfg) if bool(post_cfg.get("enabled", True)) else None

    pose_cfg = post_cfg.get("pose") or {}
    pose_estimator = None
    if post is not None:
        pose_estimator = PoseEstimator(pose_cfg)

    map_cfg = cfg.get("mapping") or {}
    mapper = BoardMapper(map_cfg) if bool(map_cfg.get("enabled", False)) else None
    occupancy_grid = OccupancyGrid()

    fuser = DetectionFuser()
    tracker = CentroidTracker()
    temporal_filter = TemporalFilter(alpha=0.7)
    counter = ObjectCounter()
    world_model = WorldModel(max_history=200)
    scene_memory = SceneMemory(max_history=50)
    perf_monitor = PerformanceMonitor()
    frame_counter = 0

    out_cfg = cfg.get("output") or {}
    save_image = bool(out_cfg.get("save_image", False))
    include_empty_frames = bool(out_cfg.get("include_empty_frames", True))
    output_format = str(out_cfg.get("format", "json")).lower()
    output_append = bool(out_cfg.get("append", True))
    FLUSH_EVERY = int(out_cfg.get("flush_every", 500))
    perf_log_interval = int(out_cfg.get("perf_log_interval", 100))
    output_history: List[Dict[str, Any]] = []
    if bool(out_cfg.get("write_outputs", True)):
        if output_format == "json":
            json_path = _resolve_path(base_dir, str(out_cfg.get("json_path", "results/output.json")))
            export_json(json_path, [], append=False)
            logger.info(f"Initialized empty JSON output file: {json_path}")
        elif output_format == "csv":
            csv_path = _resolve_path(base_dir, str(out_cfg.get("csv_path", "results/output.csv")))
            export_csv(csv_path, [], append=False)
            logger.info(f"Initialized CSV output file with header: {csv_path}")

    vis_cfg = cfg.get("visualization") or {}
    visualizer = Visualizer(vis_cfg) if bool(vis_cfg.get("enabled", True)) else None
    runtime_monitor = RuntimeMonitor() if bool(out_cfg.get("runtime_monitor", True)) else None

    counts_cfg = cfg.get("counts") or {}
    counts_enabled = bool(counts_cfg.get("enabled", True))
    counts_include = bool(counts_cfg.get("include_in_output", True))
    counts_draw = bool(counts_cfg.get("draw_on_screen", True))
    counts_position = tuple(counts_cfg.get("display_position", [10, 30]))

    show_window = bool(vis_cfg.get("show_window", True))
    window_name = str(vis_cfg.get("window_name", "AMR Vision V1"))
    if show_window and visualizer is not None:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    logger.info("pipeline started (press ESC to quit)")

    try:
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue

            ts_ms = now_ms()
            frame_counter += 1

            t0 = time.perf_counter()
            if pre is not None:
                frame = pre.process(frame)
            perf_monitor.record("preprocess", time.perf_counter() - t0)

            original_frame = frame.copy()

            t0 = time.perf_counter()
            warped_frame = perspective.apply(frame)
            perf_monitor.record("perspective", time.perf_counter() - t0)

            t0 = time.perf_counter()
            dets = detector.detect(warped_frame)
            perf_monitor.record("detect", time.perf_counter() - t0)

            t0 = time.perf_counter()
            for det in dets:
                if det.source in ["yolo", "qr", "aruco", "contour"]:
                    det.color = classifier.classify(warped_frame, det.bbox_xyxy)
            if post is not None:
                dets = post.process(dets, frame_shape_hw=warped_frame.shape[:2])
            perf_monitor.record("postprocess", time.perf_counter() - t0)

            t0 = time.perf_counter()
            det_dicts = detections_to_dicts(dets)
            det_dicts = fuser.fuse(det_dicts)
            det_dicts = tracker.update(det_dicts, timestamp_ms=ts_ms)
            perf_monitor.record("fuse+track", time.perf_counter() - t0)

            # Pose estimation for tracked objects (only if enabled)
            if pose_estimator is not None:
                det_dicts = pose_estimator.estimate_pose(det_dicts, warped_frame)

            # Pass tracker alive IDs into temporal filter to avoid aggressive pruning
            tracker_active_ids = set(getattr(tracker, "objects", {}).keys())
            det_dicts = temporal_filter.update(det_dicts, alive_ids=tracker_active_ids)

            # Update world model and scene memory with final tracked detections
            world_model.update(det_dicts)
            scene_memory.record_frame(frame_counter, det_dicts)

            # Optional mapping to grid/chess cell
            if mapper is not None:
                h, w = warped_frame.shape[:2]
                for d in det_dicts:
                    x1, y1, x2, y2 = d["bbox_xyxy"]
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    d["cell"] = mapper.pixel_to_cell(cx, cy, frame_w=w, frame_h=h)

            occupancy_data = occupancy_grid.build(det_dicts)
            errs = validate_detections(det_dicts)
            if errs:
                logger.warning(f"invalid detections: {errs[:3]}")

            count_summary = counter.update(det_dicts) if counts_enabled else {}

            payload = {
                "timestamp_ms": ts_ms,
                "frame": frame_counter,
                "image": {"width": int(warped_frame.shape[1]), "height": int(warped_frame.shape[0])},
                "detections": det_dicts,
                "detection_count": len(det_dicts),
                "detection_labels": [d["label"] for d in det_dicts],
                "object_ids": [d.get("id") for d in det_dicts if d.get("id") is not None],
                "occupancy": occupancy_data,
                "world_counts": dict(world_model.counts),
            }
            if counts_enabled and counts_include:
                payload["count_summary"] = count_summary
            # Include perf summary every N frames (not every frame to keep output lean)
            if perf_log_interval > 0 and frame_counter % perf_log_interval == 0:
                perf_summary = perf_monitor.summary()
                payload["perf_summary"] = perf_summary
                logger.info(f"[perf frame={frame_counter}] " +
                            ", ".join(f"{k}={v*1000:.1f}ms"
                                      for k, v in perf_summary.get("average_s", {}).items()))
                perf_monitor.clear()

            if bool(out_cfg.get("write_outputs", True)) and (include_empty_frames or det_dicts):
                if not output_append:
                    output_history.append(payload)
                    # Periodic flush to prevent unbounded RAM growth
                    if FLUSH_EVERY > 0 and len(output_history) >= FLUSH_EVERY:
                        export_output(output_format, out_cfg, output_history)
                        output_history.clear()
                        logger.info(f"[output] flushed {FLUSH_EVERY} frames to disk")
                else:
                    export_output(output_format, out_cfg, [payload])

            annotated = warped_frame
            if runtime_monitor is not None:
                runtime_monitor.display(det_dicts)

            if visualizer is not None:
                grid_overlay = None
                if bool(vis_cfg.get("draw_grid", True)) and mapper is not None and (cfg.get("mapping") or {}).get("grid"):
                    g = (cfg.get("mapping") or {}).get("grid") or {}
                    labels = (g.get("labels") or {}) if isinstance(g, dict) else {}
                    grid_overlay = {
                        "rows": int(g.get("rows", 8)),
                        "cols": int(g.get("cols", 8)),
                        "col_labels": str(labels.get("cols", "ABCDEFGH")),
                        "row_labels": str(labels.get("rows", "87654321")),
                    }
                annotated = visualizer.draw(
                    annotated,
                    det_dicts,
                    grid=grid_overlay,
                    occupancy=occupancy_data if occupancy_data else None,
                )

                if counts_enabled and counts_draw and count_summary:
                    annotated = visualizer.draw_counts(annotated, count_summary, position=counts_position)

                if save_image:
                    img_path = _resolve_path(base_dir, str(out_cfg.get("image_path", "results/annotated.jpg")))
                    visualizer.save(img_path, annotated)

                if show_window:
                    display_frame = annotated
                    if perspective.enabled:
                        display_frame = _side_by_side(original_frame, annotated)
                    key = visualizer.show(display_frame, window_name)
                    if key == 27:
                        break
            elif show_window:
                display_frame = annotated
                if perspective.enabled:
                    display_frame = _side_by_side(original_frame, annotated)
                cv2.imshow(window_name, display_frame)
                if (cv2.waitKey(1) & 0xFF) == 27:
                    break

    finally:
        try:
            camera.close()
        except Exception:
            pass
        try:
            detector.close()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        if bool(out_cfg.get("write_outputs", True)) and not output_append:
            if output_history:
                export_output(output_format, out_cfg, output_history)

    logger.info("shutdown complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

