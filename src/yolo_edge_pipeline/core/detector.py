"""YOLO model loading and inference utilities."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any

import numpy as np
from ultralytics import YOLO


LOGGER = logging.getLogger(__name__)


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

    detections: tuple[Detection, ...]
    annotated_frame: np.ndarray


class ObjectDetector:
    """Wrap an Ultralytics YOLO detector behind a stable project interface."""

    def __init__(
        self,
        model_path: Path,
        confidence_threshold: float = 0.25,
        device: str | None = None,
        image_size: int = 640,
    ) -> None:
        """Initialize and load the model eagerly to fail fast on bad paths."""
        self._model_path = model_path.resolve()
        self._confidence_threshold = confidence_threshold
        self._device = device
        self._image_size = image_size

        if not self._model_path.is_file():
            raise FileNotFoundError(f"YOLO model file not found: {self._model_path}")

        self._model = YOLO(str(self._model_path))
        LOGGER.info("Loaded YOLO model from %s", self._model_path)

    def predict(self, frame: np.ndarray) -> InferenceResult:
        """Run inference on a single BGR frame."""
        results = self._model.predict(
            source=frame,
            conf=self._confidence_threshold,
            imgsz=self._image_size,
            device=self._device,
            verbose=False,
        )
        result = results[0]

        detections = self._extract_detections(result)
        annotated_frame = result.plot()
        return InferenceResult(detections=detections, annotated_frame=annotated_frame)

    def _extract_detections(self, result: Any) -> tuple[Detection, ...]:
        """Convert Ultralytics results into serializable domain objects."""
        if result.boxes is None:
            return tuple()

        names = result.names
        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)

        detections = []
        for bounding_box, confidence, class_id in zip(boxes_xyxy, confidences, class_ids):
            detections.append(
                Detection(
                    class_id=int(class_id),
                    class_name=str(names[int(class_id)]),
                    confidence=float(confidence),
                    bounding_box_xyxy=tuple(float(value) for value in bounding_box),
                )
            )

        return tuple(detections)
