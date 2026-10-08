from __future__ import annotations

from unittest.mock import MagicMock
import numpy as np
import pytest

from detection.classical.ocr_detector import OCRDetector


def test_ocr_detector_mocked():
    # Setup detector
    cfg = {
        "min_confidence": 0.10,
        "min_text_length": 2,
        "min_area": 10,
        "max_aspect_ratio": 20.0,
        "tesseract_config": "--psm 11"
    }
    detector = OCRDetector(cfg)

    # Mock pytesseract calls
    mock_tesseract = MagicMock()
    mock_tesseract.image_to_string.return_value = "HELLO WORLD"
    mock_tesseract.image_to_data.return_value = {
        'text': ['HELLO', 'WORLD', ''],
        'conf': ['95.0', '98.0', '-1'],
        'left': [10, 80, 0],
        'top': [20, 20, 0],
        'width': [50, 60, 0],
        'height': [20, 20, 0]
    }
    mock_tesseract.Output.DICT = "dict"
    detector.tesseract = mock_tesseract

    # Create dummy black BGR image
    img = np.zeros((100, 200, 3), dtype=np.uint8)

    # Run detection
    dets = detector.detect(img)

    # Verify that image_to_string and image_to_data were called
    mock_tesseract.image_to_string.assert_called_once()
    mock_tesseract.image_to_data.assert_called_once()

    # Verify return values
    assert len(dets) == 2
    
    assert dets[0].label == "ocr:HELLO"
    assert dets[0].confidence == 0.95
    assert dets[0].bbox_xyxy == (10, 20, 60, 40)
    assert dets[0].source == "ocr"
    assert dets[0].meta["text"] == "HELLO"

    assert dets[1].label == "ocr:WORLD"
    assert dets[1].confidence == 0.98
    assert dets[1].bbox_xyxy == (80, 20, 140, 40)
    assert dets[1].source == "ocr"
    assert dets[1].meta["text"] == "WORLD"
