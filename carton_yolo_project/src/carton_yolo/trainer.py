"""
Purpose:
    Train the carton detection model with Ultralytics YOLO.

Why this module exists:
    Training should live in a focused module so that the main entrypoint
    remains simple and the workflow stays modular.

Resources:
    - Ultralytics train mode:
      https://docs.ultralytics.com/modes/train/
"""

from __future__ import annotations

import logging

from carton_yolo.config import AppConfig
from carton_yolo.exceptions import DatasetError

LOGGER = logging.getLogger(__name__)


def train_model(config: AppConfig) -> None:
    """Train the carton detection model using the configured dataset and settings."""
    if not config.paths.dataset_yaml_path.exists():
        raise DatasetError(
            f"Dataset YAML not found: '{config.paths.dataset_yaml_path}'. "
            "Run 'prepare_dataset' first."
        )

    from ultralytics import YOLO

    LOGGER.info("Loading model: %s", config.training.model)
    model = YOLO(config.training.model)

    project_dir = config.paths.runs_dir / config.training.project_subdir
    LOGGER.info("Starting training.")
    LOGGER.info("Project directory: %s", project_dir)
    LOGGER.info("Run name: %s", config.training.run_name)

    model.train(
        data=str(config.paths.dataset_yaml_path),
        epochs=config.training.epochs,
        imgsz=config.training.imgsz,
        batch=config.training.batch,
        workers=config.training.workers,
        patience=config.training.patience,
        device=config.training.device,
        project=str(project_dir),
        name=config.training.run_name,
        cache=config.training.cache,
        pretrained=config.training.pretrained,
        amp=config.training.amp,
        exist_ok=True,
    )

    LOGGER.info("Training completed successfully.")
