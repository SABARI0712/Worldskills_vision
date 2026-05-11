from __future__ import annotations

from typing import Any, Dict


def create_camera(cfg: Dict[str, Any]):
    source = (cfg.get("source") or "usb").lower()
    if source == "usb":
        from .usb_camera import USBCamera

        usb = cfg.get("usb", {}) or {}
        return USBCamera(
            index=int(usb.get("index", 0)),
            width=usb.get("width"),
            height=usb.get("height"),
            fps=usb.get("fps"),
            reconnect=bool(usb.get("reconnect", True)),
            reconnect_wait_s=float(usb.get("reconnect_wait_s", 0.5)),
        )

    if source == "ros":
        from .ros_camera import ROSCamera

        ros = cfg.get("ros", {}) or {}
        return ROSCamera(
            topic=str(ros.get("topic", "/camera/image_raw")),
            encoding=str(ros.get("encoding", "bgr8")),
            spin_timeout_s=float(ros.get("spin_timeout_s", 0.01)),
        )

    raise ValueError(f"unknown camera source: {source}")

