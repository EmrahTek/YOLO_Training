"""YOLO model loading and inference utilities."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any

try:
    import numpy as np
    from ultralytics import YOLO
except ModuleNotFoundError as error:
    raise ModuleNotFoundError(
        "Missing runtime dependency while importing the detector module. "
        "Activate the virtual environment and run 'pip install -r requirements.txt'."
    ) from error


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Detection:
    """Represent a single object detection."""

    class_id: int
    class_name: str
    confidence: float
    bounding_box_xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class DetectionSummary:
    """Represent aggregated detection information for one frame."""

    total_detections: int
    class_counts: dict[str, int]
    average_confidence: float

    @property
    def has_detections(self) -> bool:
        """Return True when at least one object was detected."""
        return self.total_detections > 0


@dataclass(frozen=True)
class InferenceResult:
    """Represent the structured output of one inference pass."""

    detections: tuple[Detection, ...]
    summary: DetectionSummary
    annotated_frame: np.ndarray


class ObjectDetector:
    """Wrap an Ultralytics YOLO detector behind a stable interface."""

    def __init__(
        self,
        model_path: Path,
        confidence_threshold: float = 0.25,
        device: str | None = None,
        image_size: int = 640,
    ) -> None:
        """Initialize the model eagerly so configuration errors appear early."""
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
        summary = self._summarize_detections(detections)
        annotated_frame = result.plot()
        return InferenceResult(
            detections=detections,
            summary=summary,
            annotated_frame=annotated_frame,
        )

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

    def _summarize_detections(self, detections: tuple[Detection, ...]) -> DetectionSummary:
        """Aggregate detections into compact logging-friendly statistics."""
        if not detections:
            return DetectionSummary(total_detections=0, class_counts={}, average_confidence=0.0)

        class_counts = Counter(detection.class_name for detection in detections)
        average_confidence = sum(detection.confidence for detection in detections) / len(detections)
        return DetectionSummary(
            total_detections=len(detections),
            class_counts=dict(sorted(class_counts.items())),
            average_confidence=average_confidence,
        )
