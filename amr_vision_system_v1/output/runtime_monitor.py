from typing import List, Dict, Any


class RuntimeMonitor:
    def __init__(self):
        self.last_lines: List[str] = []

    def build_lines(self, detections: List[Dict[str, Any]]) -> List[str]:
        lines: List[str] = []

        for det in detections:
            source = str(det.get("source", "UNK")).upper()
            label = det.get("label", "unknown")
            conf = det.get("confidence", 0.0)

            cell = det.get("cell", None)
            angle = det.get("angle", None)

            line = f"[{source:<6}] {label} conf={conf:.2f}"
            line += f" cell={cell}"
            if angle is not None:
                line += f" angle={round(angle, 1)}°"

            lines.append(line)

        return lines

    def display(self, detections: List[Dict[str, Any]]) -> None:
        # Disabled - using new dynamic terminal output system instead
        pass
