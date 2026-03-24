"""Dataset extraction, normalization, and validation for CVAT YOLO exports."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import random
import shutil
from typing import Iterable
import zipfile

import yaml


LOGGER = logging.getLogger(__name__)
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
SUPPORTED_LABEL_SUFFIX = ".txt"


class DatasetValidationError(ValueError):
    """Raised when a dataset cannot satisfy YOLO expectations."""


@dataclass(frozen=True)
class DatasetIssue:
    """Represent a single dataset quality issue."""

    severity: str
    message: str


@dataclass(frozen=True)
class DatasetDescription:
    """Describe the validated dataset and any discovered issues."""

    dataset_root: Path
    data_yaml_path: Path | None
    train_list_path: Path | None
    images_directory: Path | None
    labels_directory: Path | None
    class_names: dict[int, str]
    image_count: int
    label_count: int
    labeled_image_count: int
    missing_images: tuple[str, ...]
    missing_labels: tuple[str, ...]
    invalid_label_files: tuple[str, ...]
    issues: tuple[DatasetIssue, ...]

    @property
    def is_strictly_valid(self) -> bool:
        """Return True when the dataset has no blocking validation issues."""
        return not self.missing_images and not self.missing_labels


class DatasetManager:
    """Handle CVAT YOLO export directories and normalized dataset preparation."""

    def extract_cvat_zip(
        self,
        zip_path: Path,
        output_directory: Path,
        overwrite: bool = False,
    ) -> Path:
        """Extract a CVAT zip archive into a working directory."""
        zip_path = zip_path.resolve()
        output_directory = output_directory.resolve()

        if not zip_path.is_file():
            raise FileNotFoundError(f"CVAT zip file not found: {zip_path}")

        if output_directory.exists():
            if not overwrite:
                raise FileExistsError(
                    f"Output directory already exists: {output_directory}. "
                    "Pass overwrite=True to replace it."
                )
            shutil.rmtree(output_directory)

        output_directory.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zip_file:
            zip_file.extractall(output_directory)

        LOGGER.info("Extracted dataset archive to %s", output_directory)
        return output_directory

    def validate_yolo_dataset(self, dataset_root: Path) -> DatasetDescription:
        """Validate a YOLO dataset using both file-list and folder layouts."""
        dataset_root = dataset_root.resolve()
        if not dataset_root.exists():
            raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

        data_yaml_path = self._find_first_file(dataset_root, ("data.yaml", "dataset.yaml"))
        train_list_path = self._find_first_file(dataset_root, ("train.txt",))
        labels_directory = self._find_first_directory(dataset_root, ("labels",))
        images_directory = self._find_first_directory(dataset_root, ("images",))

        class_names = self._read_class_names(data_yaml_path)
        issues: list[DatasetIssue] = []
        image_stems: set[str] = set()
        missing_images: list[str] = []

        if train_list_path is not None:
            listed_image_paths = self._read_list_file(train_list_path)
            image_count = len(listed_image_paths)
            for listed_path in listed_image_paths:
                image_stems.add(Path(listed_path).stem)
                if not (dataset_root / listed_path).exists():
                    missing_images.append(listed_path)

            if missing_images:
                issues.append(
                    DatasetIssue(
                        severity="warning",
                        message=(
                            "The dataset references images that are not present in the export directory. "
                            "This is common for CVAT YOLO exports that only contain labels and train lists."
                        ),
                    )
                )
        elif images_directory is not None:
            image_files = list(self._collect_files(images_directory, SUPPORTED_IMAGE_SUFFIXES))
            image_stems = {image_path.stem for image_path in image_files}
            image_count = len(image_files)
        else:
            raise DatasetValidationError(
                "No train.txt file or images directory was found. The dataset cannot be validated."
            )

        if labels_directory is None:
            raise DatasetValidationError("No labels directory was found in the dataset.")

        label_files = list(self._collect_files(labels_directory, {SUPPORTED_LABEL_SUFFIX}))
        label_stems = {label_path.stem for label_path in label_files}
        missing_labels = sorted(image_stems - label_stems)
        label_count = len(label_files)
        invalid_label_files = self._validate_label_files(label_files, class_names)

        if missing_labels:
            issues.append(
                DatasetIssue(
                    severity="error",
                    message=(
                        f"The dataset contains {len(missing_labels)} images without labels. "
                        "Strict training should not continue until this is fixed."
                    ),
                )
            )
        if invalid_label_files:
            issues.append(
                DatasetIssue(
                    severity="error",
                    message=(
                        f"The dataset contains {len(invalid_label_files)} invalid label file(s). "
                        "Fix malformed annotations before training."
                    ),
                )
            )

        return DatasetDescription(
            dataset_root=dataset_root,
            data_yaml_path=data_yaml_path,
            train_list_path=train_list_path,
            images_directory=images_directory,
            labels_directory=labels_directory,
            class_names=class_names,
            image_count=image_count,
            label_count=label_count,
            labeled_image_count=len(sorted(image_stems & label_stems)),
            missing_images=tuple(sorted(missing_images)),
            missing_labels=tuple(missing_labels),
            invalid_label_files=tuple(invalid_label_files),
            issues=tuple(issues),
        )

    def inspect_dataset_directory(self, dataset_root: Path) -> DatasetDescription:
        """Inspect an already extracted dataset directory."""
        return self.validate_yolo_dataset(dataset_root)

    def prepare_cvat_export(
        self,
        dataset_root: Path,
        normalized_dataset_directory: Path,
        image_source_directory: Path | None = None,
        overwrite: bool = False,
        allow_missing_labels: bool = False,
    ) -> DatasetDescription:
        """Normalize an extracted CVAT YOLO export for local training or inspection."""
        description = self.validate_yolo_dataset(dataset_root)

        if description.missing_labels and not allow_missing_labels:
            raise DatasetValidationError(
                "The dataset contains missing labels. "
                "Pass allow_missing_labels=True only if you intentionally want to continue."
            )

        self._materialize_normalized_dataset(
            description=description,
            normalized_dataset_directory=normalized_dataset_directory,
            image_source_directory=image_source_directory,
            overwrite=overwrite,
        )
        return self.validate_yolo_dataset(normalized_dataset_directory)

    def create_training_dataset(
        self,
        dataset_root: Path,
        image_source_directory: Path,
        output_directory: Path,
        validation_ratio: float = 0.2,
        overwrite: bool = False,
        random_seed: int = 42,
    ) -> Path:
        """Create a clean YOLO train/val dataset using only images that have labels."""
        if not 0.0 < validation_ratio < 1.0:
            raise ValueError("validation_ratio must be between 0 and 1.")

        description = self.validate_yolo_dataset(dataset_root)
        if description.labels_directory is None:
            raise DatasetValidationError("Labels directory is required to build a training dataset.")

        image_lookup = self._build_image_lookup(image_source_directory)
        labeled_image_names = sorted(
            image_path.name
            for image_path in image_lookup.values()
            if image_path.stem not in set(description.missing_labels)
            and (description.labels_directory / "train" / f"{image_path.stem}.txt").exists()
        )

        if len(labeled_image_names) < 2:
            raise DatasetValidationError("At least two labeled images are required to build train/val splits.")

        output_directory = output_directory.resolve()
        if output_directory.exists():
            if not overwrite:
                raise FileExistsError(f"Output directory already exists: {output_directory}")
            shutil.rmtree(output_directory)

        self._create_standard_dataset_directories(output_directory)

        random_generator = random.Random(random_seed)
        shuffled_image_names = labeled_image_names[:]
        random_generator.shuffle(shuffled_image_names)

        validation_count = max(1, int(len(shuffled_image_names) * validation_ratio))
        validation_names = set(shuffled_image_names[:validation_count])

        for image_name in shuffled_image_names:
            split_name = "val" if image_name in validation_names else "train"
            source_image_path = image_lookup[image_name]
            source_label_path = description.labels_directory / "train" / f"{source_image_path.stem}.txt"

            shutil.copy2(source_image_path, output_directory / "images" / split_name / image_name)
            shutil.copy2(source_label_path, output_directory / "labels" / split_name / source_label_path.name)

        self._write_training_data_yaml(
            output_directory=output_directory,
            class_names=description.class_names,
        )
        self._write_dataset_report(
            output_directory=output_directory,
            description=description,
            train_image_names=tuple(
                image_name for image_name in shuffled_image_names if image_name not in validation_names
            ),
            validation_image_names=tuple(sorted(validation_names)),
            validation_ratio=validation_ratio,
            random_seed=random_seed,
        )
        LOGGER.info(
            "Created training dataset at %s with train=%s val=%s labeled_samples=%s",
            output_directory,
            len(shuffled_image_names) - validation_count,
            validation_count,
            len(shuffled_image_names),
        )
        return output_directory

    def _materialize_normalized_dataset(
        self,
        description: DatasetDescription,
        normalized_dataset_directory: Path,
        image_source_directory: Path | None,
        overwrite: bool,
    ) -> None:
        """Create a canonical YOLO directory structure with train image and label folders."""
        normalized_dataset_directory = normalized_dataset_directory.resolve()
        if normalized_dataset_directory.exists():
            if not overwrite:
                raise FileExistsError(
                    f"Normalized dataset directory already exists: {normalized_dataset_directory}"
                )
            shutil.rmtree(normalized_dataset_directory)

        images_train_directory = normalized_dataset_directory / "images" / "train"
        labels_train_directory = normalized_dataset_directory / "labels" / "train"
        images_train_directory.mkdir(parents=True, exist_ok=True)
        labels_train_directory.mkdir(parents=True, exist_ok=True)

        if description.labels_directory is None:
            raise DatasetValidationError("Labels directory is required to normalize the dataset.")

        for label_path in description.labels_directory.rglob("*.txt"):
            shutil.copy2(label_path, labels_train_directory / label_path.name)

        image_lookup = self._build_image_lookup(image_source_directory)
        if description.train_list_path is not None:
            for relative_image_path in self._read_list_file(description.train_list_path):
                source_image = image_lookup.get(Path(relative_image_path).name)
                if source_image is None:
                    LOGGER.warning("Skipping missing source image: %s", relative_image_path)
                    continue
                shutil.copy2(source_image, images_train_directory / source_image.name)

        data_yaml_path = normalized_dataset_directory / "data.yaml"
        yaml_content = {
            "path": str(normalized_dataset_directory),
            "train": "images/train",
            "names": description.class_names,
        }
        with data_yaml_path.open("w", encoding="utf-8") as yaml_file:
            yaml.safe_dump(yaml_content, yaml_file, sort_keys=False)

    def _create_standard_dataset_directories(self, output_directory: Path) -> None:
        """Create the standard YOLO image and label split directories."""
        for split_name in ("train", "val"):
            (output_directory / "images" / split_name).mkdir(parents=True, exist_ok=True)
            (output_directory / "labels" / split_name).mkdir(parents=True, exist_ok=True)

    def _write_training_data_yaml(self, output_directory: Path, class_names: dict[int, str]) -> None:
        """Write the dataset YAML file for YOLO training."""
        data_yaml_path = output_directory / "data.yaml"
        yaml_content = {
            "path": str(output_directory),
            "train": "images/train",
            "val": "images/val",
            "names": class_names,
        }
        with data_yaml_path.open("w", encoding="utf-8") as yaml_file:
            yaml.safe_dump(yaml_content, yaml_file, sort_keys=False)

    def _write_dataset_report(
        self,
        output_directory: Path,
        description: DatasetDescription,
        train_image_names: tuple[str, ...],
        validation_image_names: tuple[str, ...],
        validation_ratio: float,
        random_seed: int,
    ) -> None:
        """Write a dataset quality and split report for reproducibility."""
        report_path = output_directory / "dataset_report.yaml"
        report_payload = {
            "dataset_root": str(description.dataset_root),
            "image_count": description.image_count,
            "label_count": description.label_count,
            "labeled_image_count": description.labeled_image_count,
            "missing_images": list(description.missing_images),
            "missing_labels": list(description.missing_labels),
            "invalid_label_files": list(description.invalid_label_files),
            "class_names": description.class_names,
            "validation_ratio": validation_ratio,
            "random_seed": random_seed,
            "train_images": list(train_image_names),
            "validation_images": list(validation_image_names),
            "issues": [
                {"severity": issue.severity, "message": issue.message}
                for issue in description.issues
            ],
        }
        with report_path.open("w", encoding="utf-8") as report_file:
            yaml.safe_dump(report_payload, report_file, sort_keys=False)

    def _validate_label_files(self, label_files: list[Path], class_names: dict[int, str]) -> list[str]:
        """Validate YOLO label file content and return relative invalid paths."""
        invalid_label_files: list[str] = []
        known_class_ids = set(class_names.keys())
        has_declared_classes = bool(known_class_ids)

        for label_path in label_files:
            try:
                with label_path.open("r", encoding="utf-8") as label_file:
                    for line_number, line in enumerate(label_file.readlines(), start=1):
                        stripped_line = line.strip()
                        if not stripped_line:
                            continue

                        parts = stripped_line.split()
                        if len(parts) != 5:
                            raise DatasetValidationError(
                                f"Invalid annotation format in {label_path} line {line_number}: {stripped_line}"
                            )

                        class_id = int(parts[0])
                        coordinates = [float(value) for value in parts[1:]]
                        if has_declared_classes and class_id not in known_class_ids:
                            raise DatasetValidationError(
                                f"Unknown class id {class_id} in {label_path} line {line_number}"
                            )

                        if any(value < 0.0 or value > 1.0 for value in coordinates):
                            raise DatasetValidationError(
                                f"Out-of-range normalized coordinates in {label_path} line {line_number}"
                            )
            except (OSError, ValueError, DatasetValidationError):
                invalid_label_files.append(str(label_path))

        return sorted(invalid_label_files)

    def _build_image_lookup(self, image_source_directory: Path | None) -> dict[str, Path]:
        """Build an image name to path lookup from an optional local image directory."""
        if image_source_directory is None:
            return {}

        image_source_directory = image_source_directory.resolve()
        if not image_source_directory.exists():
            raise FileNotFoundError(f"Image source directory not found: {image_source_directory}")

        image_lookup: dict[str, Path] = {}
        for image_path in self._collect_files(image_source_directory, SUPPORTED_IMAGE_SUFFIXES):
            image_lookup[image_path.name] = image_path

        return image_lookup

    def _find_first_file(self, dataset_root: Path, candidate_names: tuple[str, ...]) -> Path | None:
        """Return the first matching file in the dataset tree."""
        for candidate_name in candidate_names:
            direct_match = dataset_root / candidate_name
            if direct_match.is_file():
                return direct_match.resolve()

        for candidate_path in dataset_root.rglob("*"):
            if candidate_path.is_file() and candidate_path.name.lower() in candidate_names:
                return candidate_path.resolve()

        return None

    def _find_first_directory(self, dataset_root: Path, candidate_names: tuple[str, ...]) -> Path | None:
        """Return the first matching directory in the dataset tree."""
        for candidate_name in candidate_names:
            direct_match = dataset_root / candidate_name
            if direct_match.is_dir():
                return direct_match.resolve()

        for candidate_path in dataset_root.rglob("*"):
            if candidate_path.is_dir() and candidate_path.name.lower() in candidate_names:
                return candidate_path.resolve()

        return None

    def _collect_files(self, root_directory: Path, suffixes: set[str]) -> Iterable[Path]:
        """Yield files recursively for the provided suffix set."""
        for file_path in root_directory.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in suffixes:
                yield file_path.resolve()

    def _read_class_names(self, data_yaml_path: Path | None) -> dict[int, str]:
        """Read class names from a YOLO data YAML file."""
        if data_yaml_path is None:
            return {}

        with data_yaml_path.open("r", encoding="utf-8") as yaml_file:
            yaml_content = yaml.safe_load(yaml_file) or {}

        names = yaml_content.get("names", {})
        return {int(key): str(value) for key, value in names.items()}

    def _read_list_file(self, list_file_path: Path) -> list[str]:
        """Read non-empty relative paths from a YOLO list file."""
        with list_file_path.open("r", encoding="utf-8") as list_file:
            return [line.strip() for line in list_file.readlines() if line.strip()]
