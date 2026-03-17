"""
Purpose:
    Prepare a YOLO-compatible dataset from a ready CVAT export.

Why this module exists:
    This is the bridge between annotation and training.
    It validates the CVAT export, checks image/label pairs, verifies class IDs,
    creates the train/val split, and generates the dataset YAML file.

Important assumption:
    The following folder is expected to be populated manually after CVAT export:
        data/external/cvat_export/images/
        data/external/cvat_export/labels/

Resources:
    - CVAT dataset formats:
      https://docs.cvat.ai/docs/dataset_management/formats/
    - Ultralytics detection datasets:
      https://docs.ultralytics.com/datasets/detect/
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path

import yaml

from carton_yolo.config import AppConfig
from carton_yolo.exceptions import DatasetError
from carton_yolo.utils.io_utils import clear_directory, copy_file_safe, list_files_with_extensions
from carton_yolo.utils.paths import ensure_directory

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetSummary:
    total_images: int
    train_images: int
    val_images: int


def _validate_cvat_directories(config: AppConfig) -> None:
    """Validate that the expected CVAT export folders exist."""
    if not config.paths.cvat_images_dir.exists():
        raise DatasetError(
            f"Missing CVAT images directory: '{config.paths.cvat_images_dir}'."
        )
    if not config.paths.cvat_labels_dir.exists():
        raise DatasetError(
            f"Missing CVAT labels directory: '{config.paths.cvat_labels_dir}'."
        )


def _get_image_label_pairs(config: AppConfig) -> list[tuple[Path, Path]]:
    """Return matched image/label pairs from the CVAT export directory."""
    images = list_files_with_extensions(
        config.paths.cvat_images_dir,
        config.dataset.image_extensions,
    )

    if not images:
        raise DatasetError(
            "No image files found in the CVAT export images directory."
        )

    pairs: list[tuple[Path, Path]] = []
    missing_labels: list[str] = []

    for image_path in images:
        label_path = config.paths.cvat_labels_dir / (
            image_path.stem + config.dataset.label_extension
        )
        if not label_path.exists():
            missing_labels.append(label_path.name)
            continue
        pairs.append((image_path, label_path))

    if missing_labels:
        missing_preview = ", ".join(missing_labels[:10])
        raise DatasetError(
            f"Missing label files for one or more images. Examples: {missing_preview}"
        )

    return pairs


def _validate_label_file(label_path: Path, expected_class_ids: tuple[int, ...]) -> None:
    """Validate one YOLO label file."""
    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise DatasetError(f"Unable to read label file '{label_path}'.") from exc

    if not lines:
        raise DatasetError(f"Label file is empty: '{label_path}'.")

    for line_number, line in enumerate(lines, start=1):
        parts = line.strip().split()
        if len(parts) != 5:
            raise DatasetError(
                f"Invalid YOLO label format in '{label_path}', line {line_number}."
            )

        try:
            class_id = int(float(parts[0]))
            x_center, y_center, width, height = (float(value) for value in parts[1:])
        except ValueError as exc:
            raise DatasetError(
                f"Non-numeric YOLO label value in '{label_path}', line {line_number}."
            ) from exc

        if class_id not in expected_class_ids:
            raise DatasetError(
                f"Unexpected class id {class_id} in '{label_path}', line {line_number}. "
                f"Expected only: {expected_class_ids}"
            )

        for value_name, value in {
            "x_center": x_center,
            "y_center": y_center,
            "width": width,
            "height": height,
        }.items():
            if not 0.0 <= value <= 1.0:
                raise DatasetError(
                    f"'{value_name}' out of YOLO normalized range in '{label_path}', "
                    f"line {line_number}."
                )


def _reset_output_directories(config: AppConfig) -> None:
    """Clear train and validation output directories before a fresh split."""
    for directory in [
        config.paths.dataset_images_train_dir,
        config.paths.dataset_images_val_dir,
        config.paths.dataset_labels_train_dir,
        config.paths.dataset_labels_val_dir,
        config.paths.demo_images_dir,
        config.paths.demo_labels_dir,
    ]:
        ensure_directory(directory)
        clear_directory(directory)


def _write_dataset_yaml(config: AppConfig) -> None:
    """Generate the Ultralytics dataset YAML file for the carton task."""
    yaml_content = {
        "path": str(config.paths.dataset_dir),
        "train": "images/train",
        "val": "images/val",
        "names": {index: name for index, name in enumerate(config.dataset.class_names)},
    }

    with config.paths.dataset_yaml_path.open("w", encoding="utf-8") as file_handle:
        yaml.safe_dump(yaml_content, file_handle, sort_keys=False, allow_unicode=True)


def prepare_dataset(config: AppConfig) -> DatasetSummary:
    """Build the train/val split from a ready CVAT export."""
    LOGGER.info("Starting dataset preparation.")
    _validate_cvat_directories(config)

    image_label_pairs = _get_image_label_pairs(config)
    LOGGER.info("Found %s image/label pairs in CVAT export.", len(image_label_pairs))

    for _, label_path in image_label_pairs:
        _validate_label_file(label_path, config.dataset.expected_class_ids)

    _reset_output_directories(config)

    random.seed(config.dataset.seed)
    shuffled_pairs = image_label_pairs[:]
    random.shuffle(shuffled_pairs)

    split_index = int(len(shuffled_pairs) * config.dataset.train_ratio)
    train_pairs = shuffled_pairs[:split_index]
    val_pairs = shuffled_pairs[split_index:]

    if not train_pairs or not val_pairs:
        raise DatasetError(
            "Train/validation split produced an empty subset. "
            "Add more data or adjust the ratios."
        )

    for image_path, label_path in train_pairs:
        copy_file_safe(image_path, config.paths.dataset_images_train_dir / image_path.name)
        copy_file_safe(label_path, config.paths.dataset_labels_train_dir / label_path.name)

    for image_path, label_path in val_pairs:
        copy_file_safe(image_path, config.paths.dataset_images_val_dir / image_path.name)
        copy_file_safe(label_path, config.paths.dataset_labels_val_dir / label_path.name)

    _write_dataset_yaml(config)

    LOGGER.info("Dataset preparation completed successfully.")
    LOGGER.info("Train images: %s", len(train_pairs))
    LOGGER.info("Validation images: %s", len(val_pairs))
    LOGGER.info("Dataset YAML written to: %s", config.paths.dataset_yaml_path)

    return DatasetSummary(
        total_images=len(shuffled_pairs),
        train_images=len(train_pairs),
        val_images=len(val_pairs),
    )
