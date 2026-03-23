"""Utilities for extracting and validating CVAT YOLO datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import shutil
import zipfile


SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
SUPPORTED_LABEL_SUFFIX = ".txt"


class DatasetValidationError(ValueError):
    """Raised when the dataset structure is invalid for YOLO workflows."""


@dataclass(frozen=True)
class DatasetDescription:
    """Describe the discovered YOLO dataset structure."""

    dataset_root: Path
    images_directory: Path
    labels_directory: Path
    data_yaml_path: Path | None
    image_count: int
    label_count: int


class DatasetManager:
    """Manage extraction and validation of CVAT YOLO dataset exports."""

    def extract_cvat_zip(
        self,
        zip_path: Path,
        output_directory: Path,
        overwrite: bool = False,
    ) -> Path:
        """Extract a CVAT YOLO zip archive into a target directory."""
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

            # Removing the old directory keeps the extracted dataset deterministic.
            shutil.rmtree(output_directory)

        output_directory.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as zip_file:
            zip_file.extractall(output_directory)

        dataset_root = self._resolve_dataset_root(output_directory)
        return dataset_root

    def validate_yolo_dataset(self, dataset_root: Path) -> DatasetDescription:
        """Validate a YOLO dataset and return a structured description."""
        dataset_root = dataset_root.resolve()
        if not dataset_root.exists():
            raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

        images_directory = self._find_required_directory(
            dataset_root=dataset_root,
            candidate_names=("images",),
        )
        labels_directory = self._find_required_directory(
            dataset_root=dataset_root,
            candidate_names=("labels",),
        )

        image_files = list(self._collect_files(images_directory, SUPPORTED_IMAGE_SUFFIXES))
        label_files = list(self._collect_files(labels_directory, {SUPPORTED_LABEL_SUFFIX}))

        if not image_files:
            raise DatasetValidationError("No image files were found in the dataset.")

        if not label_files:
            raise DatasetValidationError("No label files were found in the dataset.")

        self._validate_label_pairs(images_directory, labels_directory, image_files, label_files)

        return DatasetDescription(
            dataset_root=dataset_root,
            images_directory=images_directory,
            labels_directory=labels_directory,
            data_yaml_path=self.find_data_yaml(dataset_root),
            image_count=len(image_files),
            label_count=len(label_files),
        )

    def prepare_dataset(self, zip_path: Path, output_directory: Path, overwrite: bool = False) -> DatasetDescription:
        """Extract and validate a CVAT YOLO zip archive in one operation."""
        dataset_root = self.extract_cvat_zip(
            zip_path=zip_path,
            output_directory=output_directory,
            overwrite=overwrite,
        )
        return self.validate_yolo_dataset(dataset_root)

    def find_data_yaml(self, dataset_root: Path) -> Path | None:
        """Find the first data YAML file commonly used by Ultralytics."""
        candidate_files = sorted(dataset_root.rglob("*.yaml"))
        for candidate_file in candidate_files:
            if candidate_file.name.lower() in {"data.yaml", "dataset.yaml"}:
                return candidate_file.resolve()

        return candidate_files[0].resolve() if candidate_files else None

    def _resolve_dataset_root(self, extraction_directory: Path) -> Path:
        """Collapse single nested folders often found in CVAT export archives."""
        current_directory = extraction_directory.resolve()
        while True:
            child_directories = [item for item in current_directory.iterdir() if item.is_dir()]
            child_files = [item for item in current_directory.iterdir() if item.is_file()]

            if len(child_directories) != 1 or child_files:
                return current_directory

            current_directory = child_directories[0]

    def _find_required_directory(self, dataset_root: Path, candidate_names: tuple[str, ...]) -> Path:
        """Find a required directory by common YOLO dataset naming conventions."""
        for candidate_name in candidate_names:
            direct_match = dataset_root / candidate_name
            if direct_match.is_dir():
                return direct_match.resolve()

        for candidate in dataset_root.rglob("*"):
            if candidate.is_dir() and candidate.name.lower() in candidate_names:
                return candidate.resolve()

        candidate_names_text = ", ".join(candidate_names)
        raise DatasetValidationError(
            f"Could not find required dataset directory. Expected one of: {candidate_names_text}"
        )

    def _collect_files(self, root_directory: Path, suffixes: set[str]) -> Iterable[Path]:
        """Yield files recursively for the provided suffix set."""
        for file_path in root_directory.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in suffixes:
                yield file_path.resolve()

    def _validate_label_pairs(
        self,
        images_directory: Path,
        labels_directory: Path,
        image_files: list[Path],
        label_files: list[Path],
    ) -> None:
        """Ensure each image has a matching label file using relative paths."""
        image_keys = {
            image_path.relative_to(images_directory).with_suffix("").as_posix()
            for image_path in image_files
        }
        label_keys = {
            label_path.relative_to(labels_directory).with_suffix("").as_posix()
            for label_path in label_files
        }

        missing_labels = sorted(image_keys - label_keys)
        if missing_labels:
            sample = ", ".join(missing_labels[:5])
            raise DatasetValidationError(
                "Dataset validation failed because some images do not have labels. "
                f"Examples: {sample}"
            )
