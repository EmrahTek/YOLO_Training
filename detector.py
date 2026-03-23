"""YOLO model loading, inference, and annotation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from ultralytics import YOLO


@dataclass(frozen=True)
class Detection:
    """Represent a single object detection."""

    class_id: int
    class_name: str
    confidence: float
    bounding_box_xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class InferenceResult:
    """Represent the structured output of one inference pass."""

    detections: list[Detection]
    annotated_frame: np.ndarray


class ObjectDetector:
    """Wrap an Ultralytics YOLO model behind a clean application interface."""

    def __init__(self, model_path: Path, confidence_threshold: float = 0.25, device: str | None = None) -> None:
        """Initialize the detector and eagerly load the YOLO model."""
        self._model_path = model_path.resolve()
        self._confidence_threshold = confidence_threshold
        self._device = device

        if not self._model_path.is_file():
            raise FileNotFoundError(f"YOLO model file not found: {self._model_path}")

        # Eager model loading surfaces configuration issues as early as possible.
        self._model = YOLO(str(self._model_path))

    def predict(self, frame: np.ndarray) -> InferenceResult:
        """Run object detection on a single frame and return structured results."""
        results = self._model.predict(
            source=frame,
            conf=self._confidence_threshold,
            device=self._device,
            verbose=False,
        )

        result = results[0]
        detections = self._extract_detections(result)
        annotated_frame = result.plot()
        return InferenceResult(detections=detections, annotated_frame=annotated_frame)

    def _extract_detections(self, result: Any) -> list[Detection]:
        """Convert Ultralytics results into an application-friendly structure."""
        names = result.names
        detections: list[Detection] = []

        if result.boxes is None:
            return detections

        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)

        for bounding_box, confidence, class_id in zip(boxes_xyxy, confidences, class_ids):
            detections.append(
                Detection(
                    class_id=int(class_id),
                    class_name=str(names[int(class_id)]),
                    confidence=float(confidence),
                    bounding_box_xyxy=tuple(float(value) for value in bounding_box),
                )
            )

        return detections
