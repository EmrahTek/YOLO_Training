"""Configuration loading utilities for training, inference, and edge export."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = Path("configs/defaults.yaml")


@dataclass(frozen=True)
class PredictConfig:
    """Store default inference configuration values."""

    model_path: Path
    confidence: float
    device: str | None
    image_size: int
    image_directory: Path
    video_directory: Path
    output_directory: Path
    log_directory: Path
    log_level: str
    display_max_width: int
    display_max_height: int
    image_delay_ms: int
    default_camera_index: str
    external_camera_index: str


@dataclass(frozen=True)
class DatasetConfig:
    """Store dataset preparation and inspection defaults."""

    dataset_root: Path
    image_source_directory: Path
    prepared_dataset_directory: Path
    validation_ratio: float
    random_seed: int
    allow_missing_labels: bool


@dataclass(frozen=True)
class TrainConfig:
    """Store training defaults."""

    base_model: Path
    epochs: int
    image_size: int
    batch_size: int
    device: str | None
    project_directory: Path
    run_name: str
    patience: int
    workers: int
    cache: bool
    resume: bool


@dataclass(frozen=True)
class ExportConfig:
    """Store edge export defaults."""

    source_model: Path
    export_directory: Path
    formats: tuple[str, ...]
    image_size: int
    half: bool
    int8: bool
    dynamic: bool
    device: str | None
    benchmark_image: Path | None
    benchmark_runs: int


@dataclass(frozen=True)
class EvaluationConfig:
    """Store evaluation defaults."""

    model_path: Path
    dataset_yaml: Path
    split: str
    image_size: int
    batch_size: int
    device: str | None
    project_directory: Path
    run_name: str
    workers: int


@dataclass(frozen=True)
class AppConfig:
    """Store all application defaults in one structure."""

    predict: PredictConfig
    dataset: DatasetConfig
    train: TrainConfig
    export: ExportConfig
    evaluation: EvaluationConfig


def load_app_config(config_path: Path | None = None) -> AppConfig:
    """Load the project configuration from YAML or fall back to defaults."""
    normalized_path = config_path or DEFAULT_CONFIG_PATH
    yaml_content = _load_yaml(normalized_path)

    predict_section = yaml_content.get("predict", {})
    dataset_section = yaml_content.get("dataset", {})
    train_section = yaml_content.get("train", {})
    export_section = yaml_content.get("export", {})
    evaluation_section = yaml_content.get("evaluation", {})

    return AppConfig(
        predict=PredictConfig(
            model_path=Path(predict_section.get("model_path", "runs/train/carton_detector_gpu/weights/best.pt")),
            confidence=float(predict_section.get("confidence", 0.50)),
            device=_none_if_empty(predict_section.get("device")),
            image_size=int(predict_section.get("image_size", 640)),
            image_directory=Path(predict_section.get("image_directory", "data/images")),
            video_directory=Path(predict_section.get("video_directory", "data/videos")),
            output_directory=Path(predict_section.get("output_directory", "outputs")),
            log_directory=Path(predict_section.get("log_directory", "logs")),
            log_level=str(predict_section.get("log_level", "INFO")),
            display_max_width=int(predict_section.get("display_max_width", 1280)),
            display_max_height=int(predict_section.get("display_max_height", 720)),
            image_delay_ms=int(predict_section.get("image_delay_ms", 2000)),
            default_camera_index=str(predict_section.get("default_camera_index", "0")),
            external_camera_index=str(predict_section.get("external_camera_index", "1")),
        ),
        dataset=DatasetConfig(
            dataset_root=Path(dataset_section.get("dataset_root", "data/cvat_exports/caton_hause")),
            image_source_directory=Path(dataset_section.get("image_source_directory", "data/images")),
            prepared_dataset_directory=Path(dataset_section.get("prepared_dataset_directory", "data/processed/caton_hause")),
            validation_ratio=float(dataset_section.get("validation_ratio", 0.2)),
            random_seed=int(dataset_section.get("random_seed", 42)),
            allow_missing_labels=bool(dataset_section.get("allow_missing_labels", False)),
        ),
        train=TrainConfig(
            base_model=Path(train_section.get("base_model", "models/yolov8n.pt")),
            epochs=int(train_section.get("epochs", 50)),
            image_size=int(train_section.get("image_size", 640)),
            batch_size=int(train_section.get("batch_size", 8)),
            device=_none_if_empty(train_section.get("device")),
            project_directory=Path(train_section.get("project_directory", "runs/train")),
            run_name=str(train_section.get("run_name", "carton_detector_gpu")),
            patience=int(train_section.get("patience", 20)),
            workers=int(train_section.get("workers", 2)),
            cache=bool(train_section.get("cache", False)),
            resume=bool(train_section.get("resume", False)),
        ),
        export=ExportConfig(
            source_model=Path(export_section.get("source_model", "runs/train/carton_detector_gpu/weights/best.pt")),
            export_directory=Path(export_section.get("export_directory", "runs/export")),
            formats=tuple(export_section.get("formats", ["onnx", "openvino", "tflite"])),
            image_size=int(export_section.get("image_size", 640)),
            half=bool(export_section.get("half", False)),
            int8=bool(export_section.get("int8", False)),
            dynamic=bool(export_section.get("dynamic", False)),
            device=_none_if_empty(export_section.get("device")),
            benchmark_image=_optional_path(export_section.get("benchmark_image")),
            benchmark_runs=int(export_section.get("benchmark_runs", 10)),
        ),
        evaluation=EvaluationConfig(
            model_path=Path(evaluation_section.get("model_path", "runs/train/carton_detector_gpu/weights/best.pt")),
            dataset_yaml=Path(evaluation_section.get("dataset_yaml", "data/processed/caton_hause/data.yaml")),
            split=str(evaluation_section.get("split", "val")),
            image_size=int(evaluation_section.get("image_size", 640)),
            batch_size=int(evaluation_section.get("batch_size", 1)),
            device=_none_if_empty(evaluation_section.get("device")),
            project_directory=Path(evaluation_section.get("project_directory", "runs/eval")),
            run_name=str(evaluation_section.get("run_name", "carton_detector_gpu")),
            workers=int(evaluation_section.get("workers", 0)),
        ),
    )


def _load_yaml(config_path: Path) -> dict[str, Any]:
    """Load a YAML file when it exists, otherwise return an empty dictionary."""
    if not config_path.exists():
        return {}

    with config_path.open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file) or {}


def _none_if_empty(value: Any) -> str | None:
    """Normalize empty configuration values to None."""
    if value in ("", None):
        return None
    return str(value)


def _optional_path(value: Any) -> Path | None:
    """Normalize optional path values from YAML."""
    if value in ("", None):
        return None
    return Path(str(value))
