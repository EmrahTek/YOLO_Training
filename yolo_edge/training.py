"""Training workflow for the custom carton detection model."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import logging
from pathlib import Path

import yaml

from yolo_edge.config import DEFAULT_CONFIG_PATH
from yolo_edge.config import TrainConfig
from yolo_edge.config import load_app_config
from yolo_edge.data.dataset_manager import DatasetManager
from yolo_edge.utils.logging_utils import configure_logging

try:
    from ultralytics import YOLO
    import torch
except ModuleNotFoundError as error:
    raise ModuleNotFoundError(
        "Missing Ultralytics dependency. Activate the virtual environment and run 'pip install -r requirements.txt'."
    ) from error


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingRunSummary:
    """Capture the most relevant training metadata for reproducibility."""

    dataset_yaml_path: Path
    prepared_dataset_directory: Path
    base_model_path: Path
    run_directory: Path
    best_model_path: Path
    device: str
    epochs: int
    image_size: int
    batch_size: int
    patience: int
    workers: int
    cache: bool
    validation_ratio: float
    random_seed: int


def get_default_device() -> str:
    """Return the preferred training device for the current machine."""
    return "0" if torch.cuda.is_available() else "cpu"


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for training."""
    app_config = load_app_config(DEFAULT_CONFIG_PATH)
    parser = argparse.ArgumentParser(description="Prepare a custom YOLO dataset and train a carton model.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=app_config.dataset.dataset_root,
        help="Path to the extracted CVAT dataset root.",
    )
    parser.add_argument(
        "--image-source-dir",
        type=Path,
        default=app_config.dataset.image_source_directory,
        help="Directory containing the raw images referenced by the CVAT export.",
    )
    parser.add_argument(
        "--prepared-dataset-dir",
        type=Path,
        default=app_config.dataset.prepared_dataset_directory,
        help="Directory where the cleaned train/val YOLO dataset will be created.",
    )
    parser.add_argument(
        "--base-model",
        type=Path,
        default=app_config.train.base_model,
        help="Base YOLO model used as the starting point for fine-tuning.",
    )
    parser.add_argument("--epochs", type=int, default=app_config.train.epochs)
    parser.add_argument("--image-size", type=int, default=app_config.train.image_size)
    parser.add_argument("--batch-size", type=int, default=app_config.train.batch_size)
    parser.add_argument("--device", default=app_config.train.device or get_default_device())
    parser.add_argument("--validation-ratio", type=float, default=app_config.dataset.validation_ratio)
    parser.add_argument("--random-seed", type=int, default=app_config.dataset.random_seed)
    parser.add_argument("--project-dir", type=Path, default=app_config.train.project_directory)
    parser.add_argument("--run-name", default=app_config.train.run_name)
    parser.add_argument("--patience", type=int, default=app_config.train.patience)
    parser.add_argument("--workers", type=int, default=app_config.train.workers)
    parser.add_argument("--cache", action="store_true", default=app_config.train.cache)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true", default=app_config.train.resume)
    parser.add_argument("--log-level", default=app_config.predict.log_level, choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    parser.add_argument("--log-dir", type=Path, default=app_config.predict.log_directory)
    return parser


def run_training(
    dataset_root: Path,
    image_source_directory: Path,
    prepared_dataset_directory: Path,
    base_model: Path,
    epochs: int,
    image_size: int,
    batch_size: int,
    device: str,
    validation_ratio: float,
    random_seed: int,
    project_directory: Path,
    run_name: str,
    overwrite: bool,
    patience: int,
    workers: int,
    cache: bool,
    resume: bool,
) -> TrainingRunSummary:
    """Prepare the dataset, launch training, and persist training metadata."""
    _validate_training_arguments(
        validation_ratio=validation_ratio,
        epochs=epochs,
        batch_size=batch_size,
        image_size=image_size,
        patience=patience,
        workers=workers,
    )

    LOGGER.info("CUDA available=%s selected_device=%s", torch.cuda.is_available(), device)
    if torch.cuda.is_available():
        LOGGER.info("Detected GPU: %s", torch.cuda.get_device_name(0))

    dataset_manager = DatasetManager()
    description = dataset_manager.inspect_dataset_directory(dataset_root)
    LOGGER.info(
        "Dataset inspection: images=%s labels=%s labeled_images=%s missing_labels=%s invalid_labels=%s",
        description.image_count,
        description.label_count,
        description.labeled_image_count,
        len(description.missing_labels),
        len(description.invalid_label_files),
    )

    prepared_dataset_dir = dataset_manager.create_training_dataset(
        dataset_root=dataset_root,
        image_source_directory=image_source_directory,
        output_directory=prepared_dataset_directory,
        validation_ratio=validation_ratio,
        overwrite=overwrite,
        random_seed=random_seed,
    )
    data_yaml_path = prepared_dataset_dir / "data.yaml"

    LOGGER.info("Starting custom training with dataset=%s", data_yaml_path)
    model = YOLO(str(base_model.resolve()))
    training_parameters = {
        "data": str(data_yaml_path),
        "epochs": epochs,
        "imgsz": image_size,
        "batch": batch_size,
        "device": device,
        "project": str(project_directory),
        "name": run_name,
        "exist_ok": overwrite,
        "patience": patience,
        "workers": workers,
        "cache": cache,
        "seed": random_seed,
        "deterministic": True,
        "amp": torch.cuda.is_available(),
        "verbose": True,
    }
    if resume:
        training_parameters["resume"] = True
    model.train(**training_parameters)

    run_directory = project_directory / run_name
    best_model_path = run_directory / "weights" / "best.pt"
    summary = TrainingRunSummary(
        dataset_yaml_path=data_yaml_path.resolve(),
        prepared_dataset_directory=prepared_dataset_dir.resolve(),
        base_model_path=base_model.resolve(),
        run_directory=run_directory.resolve(),
        best_model_path=best_model_path.resolve(),
        device=device,
        epochs=epochs,
        image_size=image_size,
        batch_size=batch_size,
        patience=patience,
        workers=workers,
        cache=cache,
        validation_ratio=validation_ratio,
        random_seed=random_seed,
    )
    _write_training_summary(summary)
    LOGGER.info("Training finished. Best model should be available at %s", best_model_path)
    return summary


def main() -> None:
    """Prepare the dataset and train a custom YOLO model."""
    arguments = build_argument_parser().parse_args()
    configure_logging(log_level=arguments.log_level, log_directory=arguments.log_dir)

    try:
        run_training(
            dataset_root=arguments.dataset_root,
            image_source_directory=arguments.image_source_dir,
            prepared_dataset_directory=arguments.prepared_dataset_dir,
            base_model=arguments.base_model,
            epochs=arguments.epochs,
            image_size=arguments.image_size,
            batch_size=arguments.batch_size,
            device=str(arguments.device),
            validation_ratio=arguments.validation_ratio,
            random_seed=arguments.random_seed,
            project_directory=arguments.project_dir,
            run_name=arguments.run_name,
            overwrite=arguments.overwrite,
            patience=arguments.patience,
            workers=arguments.workers,
            cache=arguments.cache,
            resume=arguments.resume,
        )
    except Exception as error:
        LOGGER.exception("Training failed: %s", error)
        raise SystemExit(1) from error


def _validate_training_arguments(
    validation_ratio: float,
    epochs: int,
    batch_size: int,
    image_size: int,
    patience: int,
    workers: int,
) -> None:
    """Validate training inputs early so failures are explicit."""
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be between 0 and 1.")
    if epochs < 1:
        raise ValueError("epochs must be at least 1.")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if image_size < 32:
        raise ValueError("image_size must be at least 32.")
    if patience < 0:
        raise ValueError("patience cannot be negative.")
    if workers < 0:
        raise ValueError("workers cannot be negative.")


def _write_training_summary(summary: TrainingRunSummary) -> None:
    """Persist a compact training summary inside the run directory."""
    summary_path = summary.run_directory / "training_summary.yaml"
    summary.run_directory.mkdir(parents=True, exist_ok=True)
    payload = asdict(summary)
    payload = {key: str(value) if isinstance(value, Path) else value for key, value in payload.items()}
    with summary_path.open("w", encoding="utf-8") as summary_file:
        yaml.safe_dump(payload, summary_file, sort_keys=False)

