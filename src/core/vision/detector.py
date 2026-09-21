"""Lớp bọc bộ phát hiện đối tượng YOLOv8."""

import numpy as np
from ultralytics import YOLO

from src.core.types import Detection


class YOLODetector:
    """Wraps ultralytics YOLO for person & bag detection."""

    def __init__(
        self,
        model_path: str,
        conf: float | dict[str, float] = 0.4,
        device: str | int | None = None,
    ):
        import torch

        self.device = device if device is not None else ("0" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(model_path)
        try:
            self.model.to(self.device)
        except Exception:
            pass
        self.class_names: dict[int, str] = self.model.names
        
        # Configure thresholds per label
        if isinstance(conf, (int, float)):
            self.conf_dict: dict[str, float] = {"default": float(conf)}
            self.base_conf: float = float(conf)
        elif isinstance(conf, dict):
            self.conf_dict = conf
            self.base_conf = min(conf.values()) if conf else 0.4
        else:
            self.conf_dict = {"default": 0.4}
            self.base_conf = 0.4

    def get_threshold(self, label: str) -> float:
        """Get confidence threshold for a specific class label."""
        return self.conf_dict.get(label, self.conf_dict.get("default", self.base_conf))

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run detection on a single frame (no tracking) with per-class threshold filtering."""
        results = self.model.predict(frame, conf=self.base_conf, verbose=False, device=self.device)
        detections = []
        for r in results:
            # One device-to-host transfer for all boxes avoids a GPU sync per field.
            rows = r.boxes.data.detach().cpu().numpy()
            for row in rows:
                cls_id = int(row[-1])
                confidence = float(row[-2])
                label = self.class_names[cls_id]
                
                # Filter by class-specific threshold
                if confidence >= self.get_threshold(label):
                    detections.append(Detection(
                        bbox=tuple(float(value) for value in row[:4]),
                        confidence=confidence,
                        class_id=cls_id,
                        label=label,
                    ))
        return detections
