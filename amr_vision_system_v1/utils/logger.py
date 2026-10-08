from __future__ import annotations

import logging
import os
from datetime import datetime


def setup_logger(log_dir: str, name: str = "amr_vision_v1") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.ERROR)

    if logger.handlers:
        return logger

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"runtime_{ts}.log")

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    fh.setLevel(logging.ERROR)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    sh.setLevel(logging.ERROR)

    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.propagate = False
    return logger
