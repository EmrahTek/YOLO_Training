"""
Purpose:
    Load and validate project configuration from YAML.

Why this module exists:
    A single configuration object makes the pipeline easier to understand,
    safer to modify, and cleaner to pass into modules.

Resources:
    - PyYAML: https://pyyaml.org/wiki/PyYAMLDocumentation
    - Ultralytics docs: https://docs.ultralytics.com/
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from carton_yolo.constants import DATASET_YAML_NAME
from carton_yolo.exceptions import ConfigError


@dataclass(frozen=True)
class PathsConfig:
    project_root: Path
    cvat_export_dir: Path
    cvat_images_dir: Path
    cvat_labels_dir: Path
    dataset_dir: Path
    dataset_images_train_dir: Path
    dataset_images_val_dir: Path
    dataset_labels_train_dir: Path
    dataset_labels_val_dir: Path
    dataset_yaml_path: Path
    demo_dir: Path
    demo_images_dir: Path
    demo_labels_dir: Path
    logs_dir: Path
    runs_dir: Path
    outputs_dir: Path


@dataclass(frozen=True)
class DatasetConfig:
    seed: int
    train_ratio: float
    val_ratio: float
    image_extensions: tuple[str, ...]
    label_extension: str
    class_names: tuple[str, ...]
    expected_class_ids: tuple[int, ...]


@dataclass(frozen=True)
class TrainingConfig:
    model: str
    epochs: int
    imgsz: int
    batch: int
    workers: int
    patience: int
    device: str
    project_subdir: str
    run_name: str
    cache: bool
    pretrained: bool
    amp: bool


@dataclass(frozen=True)
class InferenceConfig:
    default_weights: Path
    conf: float
    iou: float
    imgsz: int
    device: str
    webcam_source: int
    save: bool
    save_txt: bool
    line_width: int


@dataclass(frozen=True)
class ExportConfig:
    default_weights: Path
    format: str
    imgsz: int
    dynamic: bool
    half: bool
    simplify: bool
    opset: int


@dataclass(frozen=True)
class LoggingConfig:
    level: str


@dataclass(frozen=True)
class AppConfig:
    paths: PathsConfig
    dataset: DatasetConfig
    training: TrainingConfig
    inference: InferenceConfig
    export: ExportConfig
    logging: LoggingConfig


def _load_yaml(file_path: Path) -> dict[str, Any]:
    try:
        with file_path.open("r", encoding="utf-8") as file_handle:
            content = yaml.safe_load(file_handle) or {}
    except OSError as exc:
        raise ConfigError(f"Unable to read config file: '{file_path}'.") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in config file: '{file_path}'.") from exc

    if not isinstance(content, dict):
        raise ConfigError("Root of the configuration file must be a dictionary.")

    return content


def load_app_config(config_path: Path) -> AppConfig:
    """Load application configuration from YAML."""
    raw = _load_yaml(config_path)
    project_root = config_path.resolve().parents[1]

    paths_section = raw.get("paths", {})
    dataset_section = raw.get("dataset", {})
    training_section = raw.get("training", {})
    inference_section = raw.get("inference", {})
    export_section = raw.get("export", {})
    logging_section = raw.get("logging", {})

    try:
        cvat_export_dir = (project_root / paths_section["cvat_export_dir"]).resolve()
        dataset_dir = (project_root / paths_section["dataset_dir"]).resolve()
        demo_dir = (project_root / paths_section["demo_dir"]).resolve()
        logs_dir = (project_root / paths_section["logs_dir"]).resolve()
        runs_dir = (project_root / paths_section["runs_dir"]).resolve()
        outputs_dir = (project_root / paths_section["outputs_dir"]).resolve()
    except KeyError as exc:
        raise ConfigError(f"Missing required paths configuration key: {exc}") from exc

    paths = PathsConfig(
        project_root=project_root,
        cvat_export_dir=cvat_export_dir,
        cvat_images_dir=(cvat_export_dir / "images").resolve(),
        cvat_labels_dir=(cvat_export_dir / "labels").resolve(),
        dataset_dir=dataset_dir,
        dataset_images_train_dir=(dataset_dir / "images" / "train").resolve(),
        dataset_images_val_dir=(dataset_dir / "images" / "val").resolve(),
        dataset_labels_train_dir=(dataset_dir / "labels" / "train").resolve(),
        dataset_labels_val_dir=(dataset_dir / "labels" / "val").resolve(),
        dataset_yaml_path=(dataset_dir / DATASET_YAML_NAME).resolve(),
        demo_dir=demo_dir,
        demo_images_dir=(demo_dir / "unseen_images").resolve(),
        demo_labels_dir=(demo_dir / "unseen_labels").resolve(),
        logs_dir=logs_dir,
        runs_dir=runs_dir,
        outputs_dir=outputs_dir,
    )

    try:
        dataset = DatasetConfig(
            seed=int(dataset_section["seed"]),
            train_ratio=float(dataset_section["train_ratio"]),
            val_ratio=float(dataset_section["val_ratio"]),
            image_extensions=tuple(dataset_section["image_extensions"]),
            label_extension=str(dataset_section["label_extension"]),
            class_names=tuple(dataset_section["class_names"]),
            expected_class_ids=tuple(int(value) for value in dataset_section["expected_class_ids"]),
        )
    except KeyError as exc:
        raise ConfigError(f"Missing required dataset configuration key: {exc}") from exc

    if abs((dataset.train_ratio + dataset.val_ratio) - 1.0) > 1e-9:
        raise ConfigError("Dataset train_ratio and val_ratio must sum to 1.0.")

    try:
        training = TrainingConfig(
            model=str(training_section["model"]),
            epochs=int(training_section["epochs"]),
            imgsz=int(training_section["imgsz"]),
            batch=int(training_section["batch"]),
            workers=int(training_section["workers"]),
            patience=int(training_section["patience"]),
            device=str(training_section["device"]),
            project_subdir=str(training_section["project_subdir"]),
            run_name=str(training_section["run_name"]),
            cache=bool(training_section["cache"]),
            pretrained=bool(training_section["pretrained"]),
            amp=bool(training_section["amp"]),
        )
    except KeyError as exc:
        raise ConfigError(f"Missing required training configuration key: {exc}") from exc

    try:
        inference = InferenceConfig(
            default_weights=(project_root / inference_section["default_weights"]).resolve(),
            conf=float(inference_section["conf"]),
            iou=float(inference_section["iou"]),
            imgsz=int(inference_section["imgsz"]),
            device=str(inference_section["device"]),
            webcam_source=int(inference_section["webcam_source"]),
            save=bool(inference_section["save"]),
            save_txt=bool(inference_section["save_txt"]),
            line_width=int(inference_section["line_width"]),
        )
    except KeyError as exc:
        raise ConfigError(f"Missing required inference configuration key: {exc}") from exc

    try:
        export = ExportConfig(
            default_weights=(project_root / export_section["default_weights"]).resolve(),
            format=str(export_section["format"]),
            imgsz=int(export_section["imgsz"]),
            dynamic=bool(export_section["dynamic"]),
            half=bool(export_section["half"]),
            simplify=bool(export_section["simplify"]),
            opset=int(export_section["opset"]),
        )
    except KeyError as exc:
        raise ConfigError(f"Missing required export configuration key: {exc}") from exc

    logging_cfg = LoggingConfig(level=str(logging_section.get("level", "INFO")))

    return AppConfig(
        paths=paths,
        dataset=dataset,
        training=training,
        inference=inference,
        export=export,
        logging=logging_cfg,
    )
