"""
Purpose:
    Run image, video, or webcam inference with a trained YOLO model.

Why this module exists:
    Prediction logic is isolated so it can be reused across the CLI wrappers
    and later adapted for Raspberry Pi or edge deployment.

Resources:
    - Ultralytics predict mode:
      https://docs.ultralytics.com/modes/predict/
"""

from __future__ import annotations

import logging
from pathlib import Path

from carton_yolo.config import AppConfig
from carton_yolo.exceptions import PredictionError

LOGGER = logging.getLogger(__name__)


def _load_model(weights_path: Path):
    """Load a YOLO model from a weights file."""
    from ultralytics import YOLO

    if not weights_path.exists():
        raise PredictionError(f"Model weights not found: '{weights_path}'.")
    return YOLO(str(weights_path))


def predict_images(
    config: AppConfig,
    source: Path | None = None,
    weights_path: Path | None = None,
) -> None:
    """Run inference on an image file or image directory."""
    model = _load_model(weights_path or config.inference.default_weights)
    source_path = source or config.paths.demo_images_dir

    if not source_path.exists():
        raise PredictionError(f"Image source does not exist: '{source_path}'.")

    LOGGER.info("Starting image inference on: %s", source_path)
    model.predict(
        source=str(source_path),
        conf=config.inference.conf,
        iou=config.inference.iou,
        imgsz=config.inference.imgsz,
        device=config.inference.device,
        save=config.inference.save,
        save_txt=config.inference.save_txt,
        project=str(config.paths.outputs_dir),
        name="predict_image",
        exist_ok=True,
        line_width=config.inference.line_width,
    )
    LOGGER.info("Image inference completed successfully.")


def predict_video(
    config: AppConfig,
    source: Path,
    weights_path: Path | None = None,
) -> None:
    """Run inference on a video file."""
    model = _load_model(weights_path or config.inference.default_weights)

    if not source.exists():
        raise PredictionError(f"Video source does not exist: '{source}'.")

    LOGGER.info("Starting video inference on: %s", source)
    model.predict(
        source=str(source),
        conf=config.inference.conf,
        iou=config.inference.iou,
        imgsz=config.inference.imgsz,
        device=config.inference.device,
        save=config.inference.save,
        save_txt=config.inference.save_txt,
        project=str(config.paths.outputs_dir),
        name="predict_video",
        exist_ok=True,
        line_width=config.inference.line_width,
    )
    LOGGER.info("Video inference completed successfully.")


def predict_camera(
    config: AppConfig,
    camera_index: int | None = None,
    weights_path: Path | None = None,
) -> None:
    """Run real-time inference from a webcam source."""
    model = _load_model(weights_path or config.inference.default_weights)
    webcam_source = camera_index if camera_index is not None else config.inference.webcam_source

    LOGGER.info("Starting webcam inference on source: %s", webcam_source)
    model.predict(
        source=webcam_source,
        conf=config.inference.conf,
        iou=config.inference.iou,
        imgsz=config.inference.imgsz,
        device=config.inference.device,
        save=config.inference.save,
        save_txt=config.inference.save_txt,
        project=str(config.paths.outputs_dir),
        name="predict_camera",
        exist_ok=True,
        line_width=config.inference.line_width,
        stream=False,
        show=True,
    )
    LOGGER.info("Webcam inference completed successfully.")
