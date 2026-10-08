from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from detection.hybrid_detector import HybridDetector
from detection.types import Detection
from output.visualizer import Visualizer


class TestPerceptionOptimizations(unittest.TestCase):

    def test_visualizer_static_caching(self):
        # 1. Initialize Visualizer
        cfg = {"line_thickness": 3, "font_scale": 0.75, "tag_thickness": 2}
        vis = Visualizer(cfg)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        grid = {"rows": 4, "cols": 4, "col_labels": "ABCD", "row_labels": "8765"}

        # First draw: Pre-render should trigger
        assert vis._static_grid_overlay is None
        out1 = vis.draw(frame, [], grid=grid)
        assert vis._static_grid_overlay is not None
        
        # Capture the pre-rendered overlay instance
        first_overlay = vis._static_grid_overlay

        # Second draw: Pre-render should be reused
        out2 = vis.draw(frame, [], grid=grid)
        assert vis._static_grid_overlay is first_overlay

    def test_detector_scheduling_and_cache(self):
        # 2. Setup mock detectors
        mock_yolo = MagicMock()
        mock_qr = MagicMock()
        mock_yolo.detect.return_value = [
            Detection(label="yolo:cup", confidence=0.9, bbox_xyxy=(50, 50, 150, 150), source="yolo", meta={})
        ]
        mock_qr.detect.return_value = []

        cfg = {
            "mode": "hybrid",
            "confidence": 0.5,
            "yolo": {"enabled": True},
            "classical": {
                "enabled": True,
                "qr": {"enabled": True},
                "aruco": {"enabled": False},
                "ocr": {"enabled": False},
                "color": {"enabled": False},
                "contour": {"enabled": False},
            }
        }

        detector = HybridDetector(cfg)
        detector._yolo = mock_yolo
        detector._qr = mock_qr

        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Frame 1: index = 1.
        # run_yolo = (1 % 3 == 0) -> False (skipped, uses empty cache)
        # run_qr = (1 % 2 == 0) -> False (skipped, uses empty cache)
        dets1 = detector.detect(frame)
        assert len(dets1) == 0
        mock_yolo.detect.assert_not_called()
        mock_qr.detect.assert_not_called()

        # Frame 2: index = 2.
        # run_yolo = (2 % 3 == 0) -> False (skipped)
        # run_qr = (2 % 2 == 0) -> True (executed)
        dets2 = detector.detect(frame)
        mock_qr.detect.assert_called_once()
        mock_yolo.detect.assert_not_called()

        # Frame 3: index = 3.
        # run_yolo = (3 % 3 == 0) -> True (executed)
        # run_qr = (3 % 2 == 0) -> False (skipped, carries over previous QR empty cache)
        dets3 = detector.detect(frame)
        mock_yolo.detect.assert_called_once()
        assert len(dets3) == 1
        assert dets3[0].label == "yolo:cup"

        # Frame 4: index = 4.
        # run_yolo = (4 % 3 == 0) -> False (skipped, carries over cup!)
        # run_qr = (4 % 2 == 0) -> True (executed)
        dets4 = detector.detect(frame)
        assert len(dets4) == 1
        assert dets4[0].label == "yolo:cup"

    def test_roi_coordinate_mapping(self):
        # 3. Test that cropped ROI QR detections shift correctly back to global coordinates
        mock_yolo = MagicMock()
        mock_qr = MagicMock()
        
        # Bounding box coordinates
        yolo_box = (100, 100, 300, 300)
        mock_yolo.detect.return_value = [
            Detection(label="yolo:label", confidence=0.9, bbox_xyxy=yolo_box, source="yolo", meta={})
        ]
        # QR detected within crop at (10, 10, 50, 50)
        mock_qr.detect.return_value = [
            Detection(label="qr:ID123", confidence=1.0, bbox_xyxy=(10, 10, 50, 50), source="qr", meta={})
        ]

        cfg = {
            "mode": "hybrid",
            "confidence": 0.5,
            "yolo": {"enabled": True},
            "classical": {
                "enabled": True,
                "qr": {"enabled": True},
                "aruco": {"enabled": False},
                "ocr": {"enabled": False},
            }
        }
        detector = HybridDetector(cfg)
        detector._yolo = mock_yolo
        detector._qr = mock_qr

        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Trigger Frame 6 (index = 6, both run_yolo and run_qr are True)
        detector._frame_index = 5
        dets = detector.detect(frame)

        # Find QR detection
        qr_det = next(d for d in dets if d.source == "qr")
        # Global coordinates should be: crop coordinates + crop origin (which is box coordinates - 10 pad)
        # Pad is 10, so origin is x=90, y=90.
        # Shifted coordinates: x1 = 10+90=100, y1 = 10+90=100, x2 = 50+90=140, y2 = 50+90=140.
        assert qr_det.bbox_xyxy == (100, 100, 140, 140)
