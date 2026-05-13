from __future__ import annotations

import os
import sys
from typing import Any, Dict, List

import cv2
import numpy as np

from config.loader import load_config
from perception.occupancy_grid import OccupancyGrid
from camera.camera_handler import create_camera
from preprocessing.image_preprocessor import ImagePreprocessor
from detection.hybrid_detector import HybridDetector
from detection.types import detections_to_dicts
from postprocessing.post_processor import PostProcessor
from postprocessing.board_mapper import BoardMapper
from postprocessing.perspective_transform import PerspectiveTransform
from postprocessing.pose_estimator import PoseEstimator
from output.formatter import export_output
from output.visualizer import Visualizer
from fusion.detection_fuser import DetectionFuser
from fusion.duplicate_resolver import DuplicateResolver
from tracking.centroid_tracker import CentroidTracker
from tracking.temporal_filter import TemporalFilter
from perception.color_classifier import ColorClassifier
from utils.helpers import now_ms
from utils.logger import setup_logger
from utils.validators import validate_detections


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
    pose_estimator = PoseEstimator(pose_cfg)

    map_cfg = cfg.get("mapping") or {}
    mapper = BoardMapper(map_cfg) if bool(map_cfg.get("enabled", False)) else None
    occupancy_grid = OccupancyGrid()

    resolver = DuplicateResolver()
    fuser = DetectionFuser()
    tracker = CentroidTracker()
    temporal_filter = TemporalFilter(alpha=0.7)

    out_cfg = cfg.get("output") or {}
    save_image = bool(out_cfg.get("save_image", False))
    include_empty_frames = bool(out_cfg.get("include_empty_frames", True))
    output_format = str(out_cfg.get("format", "json")).lower()
    output_append = bool(out_cfg.get("append", False))
    output_history: List[Dict[str, Any]] = []
    vis_cfg = cfg.get("visualization") or {}
    visualizer = Visualizer(vis_cfg) if bool(vis_cfg.get("enabled", True)) else None

    show_window = bool(vis_cfg.get("show_window", True))
    window_name = str(vis_cfg.get("window_name", "AMR Vision V1"))
    if show_window and visualizer is not None:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    logger.info("pipeline started (press ESC to quit)")

    try:
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                continue

            ts_ms = now_ms()
            if pre is not None:
                frame = pre.process(frame)

            original_frame = frame.copy()
            warped_frame = perspective.apply(frame)

            dets = detector.detect(warped_frame)
            for det in dets:
                if det.source in ["yolo", "qr", "aruco", "contour"]:
                    det.color = classifier.classify(warped_frame, det.bbox_xyxy)
            if post is not None:
                dets = post.process(dets, frame_shape_hw=warped_frame.shape[:2])

            det_dicts = detections_to_dicts(dets)
            det_dicts = fuser.fuse(det_dicts)
            det_dicts = tracker.update(det_dicts)

            # Pose estimation for tracked objects
            det_dicts = pose_estimator.estimate_pose(det_dicts, warped_frame)
            det_dicts = temporal_filter.update(det_dicts)

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

            payload = {
                "timestamp_ms": ts_ms,
                "image": {"width": int(warped_frame.shape[1]), "height": int(warped_frame.shape[0])},
                "detections": det_dicts,
                "detection_count": len(det_dicts),
                "detection_labels": [d["label"] for d in det_dicts],
                "object_ids": [d.get("id") for d in det_dicts if d.get("id") is not None],
                "occupancy": occupancy_data,
            }

            if bool(out_cfg.get("write_outputs", True)) and (include_empty_frames or det_dicts):
                if not output_append:
                    # Buffer frames when not appending (for JSON/CSV)
                    output_history.append(payload)
                else:
                    # Write immediately in append mode (JSONL/CSV append)
                    export_output(output_format, out_cfg, payload)

            annotated = warped_frame
            if visualizer is not None:
                grid_overlay = None
                if bool(vis_cfg.get("draw_grid", True)) and (cfg.get("mapping") or {}).get("grid"):
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

