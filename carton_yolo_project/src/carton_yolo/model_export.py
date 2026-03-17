"""
Purpose:
    Export a trained model into a deployment-friendly format.

Why this module exists:
    Export is a separate pipeline stage. Keeping it isolated makes later
    Raspberry Pi or edge deployment work easier.

Resources:
    - Ultralytics export mode:
      https://docs.ultralytics.com/modes/export/
"""

from __future__ import annotations

import logging
from pathlib import Path

from carton_yolo.config import AppConfig
from carton_yolo.exceptions import PredictionError

LOGGER = logging.getLogger(__name__)


def export_model(config: AppConfig, weights_path: Path | None = None) -> None:
    """Export the trained model into the configured deployment format."""
    from ultralytics import YOLO

    resolved_weights = weights_path or config.export.default_weights
    if not resolved_weights.exists():
        raise PredictionError(
            f"Export weights not found: '{resolved_weights}'."
        )

    LOGGER.info("Loading weights for export: %s", resolved_weights)
    model = YOLO(str(resolved_weights))

    LOGGER.info("Starting model export with format: %s", config.export.format)
    model.export(
        format=config.export.format,
        imgsz=config.export.imgsz,
        dynamic=config.export.dynamic,
        half=config.export.half,
        simplify=config.export.simplify,
        opset=config.export.opset,
    )
    LOGGER.info("Model export completed successfully.")
