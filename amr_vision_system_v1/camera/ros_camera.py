from __future__ import annotations

import threading
from typing import Optional


class ROSCamera:
    """
    ROS2 image topic camera.

    This module is imported only when `camera.source: ros` is selected.
    """

    def __init__(self, topic: str, encoding: str = "bgr8", spin_timeout_s: float = 0.01) -> None:
        self.topic = topic
        self.encoding = encoding
        self.spin_timeout_s = spin_timeout_s

        self._lock = threading.Lock()
        self._frame = None

        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from sensor_msgs.msg import Image
        from cv_bridge import CvBridge

        if not rclpy.ok():
            rclpy.init()

        self._rclpy = rclpy
        self._executor = SingleThreadedExecutor()
        self._bridge = CvBridge()

        class _Node(Node):
            pass

        self._node = _Node("amr_vision_ros_camera")
        self._executor.add_node(self._node)

        def _cb(msg: Image) -> None:
            try:
                frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding=self.encoding)
                with self._lock:
                    self._frame = frame
            except Exception:
                return

        self._node.create_subscription(Image, self.topic, _cb, 10)

    def read(self):
        self._executor.spin_once(timeout_sec=float(self.spin_timeout_s))
        with self._lock:
            if self._frame is None:
                return False, None
            return True, self._frame.copy()

    def close(self) -> None:
        try:
            self._executor.remove_node(self._node)
            self._node.destroy_node()
        except Exception:
            pass
        try:
            if self._rclpy.ok():
                self._rclpy.shutdown()
        except Exception:
            pass

