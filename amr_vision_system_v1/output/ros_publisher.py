from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional


class ROSPublisherManager:
    """
    ROS2 publisher manager for publishing clean detection data to dedicated topics:
    - YOLO -> /camera/object
    - Color -> /camera/colour
    - QR -> /camera/qr
    - ArUco -> /camera/aruco
    - OCR -> /camera/ocr
    """

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger("amr_vision_system")
        self.node = None
        self.publishers = {}
        self._String = None

        try:
            import rclpy
            from std_msgs.msg import String
            self._String = String

            if not rclpy.ok():
                rclpy.init()

            from rclpy.node import Node
            self.node = Node("amr_vision_publisher")

            # Initialize publishers for each detection source
            self.publishers = {
                "yolo": self.node.create_publisher(String, "/camera/object", 10),
                "color": self.node.create_publisher(String, "/camera/colour", 10),
                "qr": self.node.create_publisher(String, "/camera/qr", 10),
                "aruco": self.node.create_publisher(String, "/camera/aruco", 10),
                "ocr": self.node.create_publisher(String, "/camera/ocr", 10),
            }
            self.logger.info("ROS2 publishers successfully initialized.")
        except Exception as e:
            self.logger.warning(
                f"ROS2 not available or failed to initialize publishers: {e}"
            )

    def publish_detections(self, det_dicts: List[Dict[str, Any]]) -> None:
        """
        Publish the data alone (name/text/payload) to the respective ROS2 topic.
        """
        if not self.node or not self.publishers or not self._String:
            return

        for det in det_dicts:
            source = str(det.get("source", "")).lower()
            # Normalize source names
            if source == "colour":
                source = "color"

            pub = self.publishers.get(source)
            if pub:
                try:
                    label = str(det.get("label", ""))
                    # Extract raw data / clean name alone (remove prefix if present)
                    if ":" in label:
                        data = label.split(":", 1)[1]
                    else:
                        data = label

                    msg = self._String()
                    msg.data = data
                    pub.publish(msg)
                except Exception as e:
                    self.logger.error(
                        f"Failed to publish to ROS2 topic for source {source}: {e}"
                    )
