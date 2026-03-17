"""
Purpose:
    Run validation on the trained carton model.

Why this module exists:
    Validation should be callable independently from training so that
    reproducibility and experimentation stay clean.

Resources:
    - Ultralytics validation:
      https://docs.ultralytics.com/modes/val/
"""

from __future__ import annotations

import logging
from pathlib import Path

from carton_yolo.config import AppConfig
from carton_yolo.exceptions import PredictionError

LOGGER = logging.getLogger(__name__)


def validate_model(config: AppConfig, weights_path: Path | None = None) -> None:
    """Validate a trained model on the configured validation set."""
    from ultralytics import YOLO

    resolved_weights = weights_path or config.inference.default_weights
    if not resolved_weights.exists():
        raise PredictionError(
            f"Validation weights not found: '{resolved_weights}'."
        )

    LOGGER.info("Loading weights for validation: %s", resolved_weights)
    model = YOLO(str(resolved_weights))

    LOGGER.info("Starting validation.")
    model.val(
        data=str(config.paths.dataset_yaml_path),
        imgsz=config.inference.imgsz,
        conf=config.inference.conf,
        iou=config.inference.iou,
        device=config.inference.device,
    )
    LOGGER.info("Validation completed successfully.")
