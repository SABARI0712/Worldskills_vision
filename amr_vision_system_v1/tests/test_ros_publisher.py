from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch
import pytest

from output.ros_publisher import ROSPublisherManager


def test_ros_publisher_manager():
    # Setup mocks for ROS2 dependencies
    mock_node = MagicMock()
    mock_pub_yolo = MagicMock()
    mock_pub_color = MagicMock()
    mock_pub_qr = MagicMock()
    mock_pub_aruco = MagicMock()
    mock_pub_ocr = MagicMock()
    
    mock_node.create_publisher.side_effect = lambda msg_type, topic, qos: {
        "/camera/object": mock_pub_yolo,
        "/camera/colour": mock_pub_color,
        "/camera/qr": mock_pub_qr,
        "/camera/aruco": mock_pub_aruco,
        "/camera/ocr": mock_pub_ocr,
    }[topic]

    # Patch rclpy and String import during initialization
    with patch("builtins.__import__") as mock_import:
        # Mock sys.modules for rclpy and std_msgs.msg
        mock_rclpy = MagicMock()
        mock_rclpy.ok.return_value = True
        
        sys.modules["rclpy"] = mock_rclpy
        sys.modules["rclpy.node"] = MagicMock()
        sys.modules["std_msgs"] = MagicMock()
        sys.modules["std_msgs.msg"] = MagicMock()
        
        # Use a simple dummy class to prevent mock sharing side-effects
        class DummyString:
            def __init__(self):
                self.data = ""
        
        # Instantiate publisher manager
        manager = ROSPublisherManager()
        manager.node = mock_node
        manager._String = DummyString
        manager.publishers = {
            "yolo": mock_pub_yolo,
            "color": mock_pub_color,
            "qr": mock_pub_qr,
            "aruco": mock_pub_aruco,
            "ocr": mock_pub_ocr,
        }

        # Setup test detections (with standard source prefixes)
        det_dicts = [
            {"source": "yolo", "label": "yolo:cup"},
            {"source": "color", "label": "colour:red"},
            {"source": "qr", "label": "qr:https://google.com"},
            {"source": "aruco", "label": "aruco:42"},
            {"source": "ocr", "label": "ocr:HELLO"},
            {"source": "unknown", "label": "unknown:val"}
        ]

        # Trigger publishing
        manager.publish_detections(det_dicts)

        # Verify yolo -> /camera/object (data alone = "cup")
        mock_pub_yolo.publish.assert_called_once()
        sent_yolo = mock_pub_yolo.publish.call_args[0][0]
        assert sent_yolo.data == "cup"

        # Verify color -> /camera/colour (data alone = "red")
        mock_pub_color.publish.assert_called_once()
        sent_color = mock_pub_color.publish.call_args[0][0]
        assert sent_color.data == "red"

        # Verify qr -> /camera/qr (data alone = "https://google.com")
        mock_pub_qr.publish.assert_called_once()
        sent_qr = mock_pub_qr.publish.call_args[0][0]
        assert sent_qr.data == "https://google.com"

        # Verify aruco -> /camera/aruco (data alone = "42")
        mock_pub_aruco.publish.assert_called_once()
        sent_aruco = mock_pub_aruco.publish.call_args[0][0]
        assert sent_aruco.data == "42"

        # Verify ocr -> /camera/ocr (data alone = "HELLO")
        mock_pub_ocr.publish.assert_called_once()
        sent_ocr = mock_pub_ocr.publish.call_args[0][0]
        assert sent_ocr.data == "HELLO"
