"""Training workflow for the custom carton detection model."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from yolo_edge.data.dataset_manager import DatasetManager
from yolo_edge.utils.logging_utils import configure_logging

try:
    from ultralytics import YOLO
except ModuleNotFoundError as error:
    raise ModuleNotFoundError(
        "Missing Ultralytics dependency. Activate the virtual environment and run 'pip install -r requirements.txt'."
    ) from error


LOGGER = logging.getLogger(__name__)


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for training."""
    parser = argparse.ArgumentParser(description="Prepare a custom YOLO dataset and train a carton model.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/cvat_exports/caton_hause"),
        help="Path to the extracted CVAT dataset root.",
    )
    parser.add_argument(
        "--image-source-dir",
        type=Path,
        default=Path("data/images"),
        help="Directory containing the raw images referenced by the CVAT export.",
    )
    parser.add_argument(
        "--prepared-dataset-dir",
        type=Path,
        default=Path("data/processed/caton_hause"),
        help="Directory where the cleaned train/val YOLO dataset will be created.",
    )
    parser.add_argument(
        "--base-model",
        type=Path,
        default=Path("models/yolov8n.pt"),
        help="Base YOLO model used as the starting point for fine-tuning.",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--project-dir", type=Path, default=Path("runs/train"))
    parser.add_argument("--run-name", default="carton_detector")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    return parser


def main() -> None:
    """Prepare the dataset and train a custom YOLO model."""
    arguments = build_argument_parser().parse_args()
    configure_logging(log_level=arguments.log_level, log_directory=arguments.log_dir)

    try:
        dataset_manager = DatasetManager()
        prepared_dataset_dir = dataset_manager.create_training_dataset(
            dataset_root=arguments.dataset_root,
            image_source_directory=arguments.image_source_dir,
            output_directory=arguments.prepared_dataset_dir,
            validation_ratio=arguments.validation_ratio,
            overwrite=arguments.overwrite,
        )
        data_yaml_path = prepared_dataset_dir / "data.yaml"

        LOGGER.info("Starting custom training with dataset=%s", data_yaml_path)
        model = YOLO(str(arguments.base_model.resolve()))
        training_results = model.train(
            data=str(data_yaml_path),
            epochs=arguments.epochs,
            imgsz=arguments.image_size,
            batch=arguments.batch_size,
            device=arguments.device,
            project=str(arguments.project_dir),
            name=arguments.run_name,
            exist_ok=arguments.overwrite,
        )

        best_model_path = arguments.project_dir / arguments.run_name / "weights" / "best.pt"
        LOGGER.info("Training finished. Best model should be available at %s", best_model_path)
        LOGGER.debug("Training results object: %s", training_results)
    except Exception as error:
        LOGGER.exception("Training failed: %s", error)
        raise SystemExit(1) from error
