"""Evaluation workflow for trained YOLO carton detection models."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import yaml

from yolo_edge.config import DEFAULT_CONFIG_PATH
from yolo_edge.config import load_app_config
from yolo_edge.utils.logging_utils import configure_logging

try:
    from ultralytics import YOLO
    import torch
except ModuleNotFoundError as error:
    raise ModuleNotFoundError(
        "Missing Ultralytics dependency. Activate the virtual environment and run 'pip install -r requirements.txt'."
    ) from error


LOGGER = logging.getLogger(__name__)
BACKGROUND_LABEL = "background"


@dataclass(frozen=True)
class ClassEvaluationSummary:
    """Store the most important metrics for one class."""

    class_index: int
    class_name: str
    support: int
    present_in_split: bool
    precision: float | None
    recall: float | None
    f1_score: float | None
    map50: float | None
    map50_95: float | None


@dataclass(frozen=True)
class EvaluationRunSummary:
    """Capture a reproducible evaluation summary."""

    model_path: Path
    dataset_yaml_path: Path
    save_directory: Path
    split: str
    device: str
    image_size: int
    batch_size: int
    workers: int
    instance_count: int
    precision: float
    recall: float
    f1_score: float
    map50: float
    map50_95: float
    confusion_matrix_csv_path: Path
    normalized_confusion_matrix_csv_path: Path
    confusion_matrix_image_path: Path
    normalized_confusion_matrix_image_path: Path
    class_metrics: tuple[ClassEvaluationSummary, ...]


def get_default_device() -> str:
    """Return the preferred evaluation device for the current machine."""
    return "0" if torch.cuda.is_available() else "cpu"


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for model evaluation."""
    app_config = load_app_config(DEFAULT_CONFIG_PATH)
    parser = argparse.ArgumentParser(description="Evaluate a trained YOLO model on the prepared dataset.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--model-path", type=Path, default=app_config.evaluation.model_path)
    parser.add_argument("--dataset-yaml", type=Path, default=app_config.evaluation.dataset_yaml)
    parser.add_argument("--split", choices=("train", "val", "test"), default=app_config.evaluation.split)
    parser.add_argument("--image-size", type=int, default=app_config.evaluation.image_size)
    parser.add_argument("--batch-size", type=int, default=app_config.evaluation.batch_size)
    parser.add_argument("--device", default=app_config.evaluation.device or get_default_device())
    parser.add_argument("--project-dir", type=Path, default=app_config.evaluation.project_directory)
    parser.add_argument("--run-name", default=app_config.evaluation.run_name)
    parser.add_argument("--workers", type=int, default=app_config.evaluation.workers)
    parser.add_argument("--log-level", default=app_config.predict.log_level, choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    parser.add_argument("--log-dir", type=Path, default=app_config.predict.log_directory)
    return parser


def run_evaluation(
    model_path: Path,
    dataset_yaml_path: Path,
    split: str,
    image_size: int,
    batch_size: int,
    device: str,
    project_directory: Path,
    run_name: str | None,
    workers: int,
) -> EvaluationRunSummary:
    """Run validation, persist artifacts, and return a compact metric summary."""
    _validate_evaluation_arguments(
        split=split,
        image_size=image_size,
        batch_size=batch_size,
        workers=workers,
    )

    normalized_model_path = model_path.resolve()
    normalized_dataset_yaml_path = dataset_yaml_path.resolve()
    normalized_project_directory = project_directory.resolve()
    resolved_run_name = run_name or derive_default_run_name(normalized_model_path)
    resolved_device = str(device)

    if not normalized_model_path.is_file():
        raise FileNotFoundError(f"Model weights not found: {normalized_model_path}")
    if not normalized_dataset_yaml_path.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {normalized_dataset_yaml_path}")

    _configure_matplotlib_cache()
    LOGGER.info(
        "Starting evaluation with model=%s dataset=%s split=%s device=%s",
        normalized_model_path,
        normalized_dataset_yaml_path,
        split,
        resolved_device,
    )

    model = YOLO(str(normalized_model_path))
    metrics = model.val(
        data=str(normalized_dataset_yaml_path),
        split=split,
        imgsz=image_size,
        batch=batch_size,
        device=resolved_device,
        project=str(normalized_project_directory),
        name=resolved_run_name,
        exist_ok=True,
        workers=workers,
        save_json=False,
        plots=True,
        verbose=True,
    )

    save_directory = Path(metrics.save_dir).resolve()
    confusion_matrix = np.asarray(metrics.confusion_matrix.matrix, dtype=float)
    normalized_confusion_matrix = normalize_confusion_matrix(confusion_matrix)
    class_labels = [str(name) for _, name in sorted(metrics.names.items())] + [BACKGROUND_LABEL]

    confusion_matrix_csv_path = save_directory / "confusion_matrix.csv"
    normalized_confusion_matrix_csv_path = save_directory / "confusion_matrix_normalized.csv"
    confusion_matrix_image_path = save_directory / "confusion_matrix.png"
    normalized_confusion_matrix_image_path = save_directory / "confusion_matrix_normalized.png"

    write_confusion_matrix_csv(
        output_path=confusion_matrix_csv_path,
        class_labels=class_labels,
        matrix=confusion_matrix,
    )
    write_confusion_matrix_csv(
        output_path=normalized_confusion_matrix_csv_path,
        class_labels=class_labels,
        matrix=normalized_confusion_matrix,
    )

    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)
    summary = EvaluationRunSummary(
        model_path=normalized_model_path,
        dataset_yaml_path=normalized_dataset_yaml_path,
        save_directory=save_directory,
        split=split,
        device=resolved_device,
        image_size=image_size,
        batch_size=batch_size,
        workers=workers,
        instance_count=int(np.asarray(metrics.nt_per_class).sum()),
        precision=precision,
        recall=recall,
        f1_score=calculate_f1_score(precision, recall),
        map50=float(metrics.box.map50),
        map50_95=float(metrics.box.map),
        confusion_matrix_csv_path=confusion_matrix_csv_path,
        normalized_confusion_matrix_csv_path=normalized_confusion_matrix_csv_path,
        confusion_matrix_image_path=confusion_matrix_image_path,
        normalized_confusion_matrix_image_path=normalized_confusion_matrix_image_path,
        class_metrics=build_class_summaries(metrics),
    )
    write_evaluation_summary(summary)
    LOGGER.info("Evaluation finished. Summary available at %s", save_directory / "evaluation_summary.yaml")
    return summary


def main() -> None:
    """Evaluate a trained YOLO model and persist its metrics."""
    arguments = build_argument_parser().parse_args()
    configure_logging(log_level=arguments.log_level, log_directory=arguments.log_dir)

    try:
        run_evaluation(
            model_path=arguments.model_path,
            dataset_yaml_path=arguments.dataset_yaml,
            split=arguments.split,
            image_size=arguments.image_size,
            batch_size=arguments.batch_size,
            device=str(arguments.device),
            project_directory=arguments.project_dir,
            run_name=arguments.run_name,
            workers=arguments.workers,
        )
    except Exception as error:
        LOGGER.exception("Evaluation failed: %s", error)
        raise SystemExit(1) from error


def calculate_f1_score(precision: float, recall: float) -> float:
    """Calculate the harmonic mean of precision and recall."""
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def derive_default_run_name(model_path: Path) -> str:
    """Infer a stable evaluation run name from a model path."""
    if model_path.parent.name == "weights" and len(model_path.parents) >= 2:
        return model_path.parents[1].name
    return f"{model_path.stem}_evaluation"


def build_class_summaries(metrics: Any) -> tuple[ClassEvaluationSummary, ...]:
    """Convert class-wise Ultralytics metrics into a stable serializable form."""
    metric_indices = {
        int(class_index): metric_index
        for metric_index, class_index in enumerate(np.asarray(metrics.ap_class_index).tolist())
    }
    support_counts = np.asarray(metrics.nt_per_class).astype(int).tolist()
    summaries: list[ClassEvaluationSummary] = []

    for class_index, class_name in sorted(metrics.names.items()):
        support = support_counts[class_index] if class_index < len(support_counts) else 0
        metric_index = metric_indices.get(class_index)

        if metric_index is None:
            summaries.append(
                ClassEvaluationSummary(
                    class_index=int(class_index),
                    class_name=str(class_name),
                    support=int(support),
                    present_in_split=False,
                    precision=None,
                    recall=None,
                    f1_score=None,
                    map50=None,
                    map50_95=None,
                )
            )
            continue

        precision, recall, map50, map50_95 = metrics.box.class_result(metric_index)
        numeric_precision = float(precision)
        numeric_recall = float(recall)
        summaries.append(
            ClassEvaluationSummary(
                class_index=int(class_index),
                class_name=str(class_name),
                support=int(support),
                present_in_split=True,
                precision=numeric_precision,
                recall=numeric_recall,
                f1_score=calculate_f1_score(numeric_precision, numeric_recall),
                map50=float(map50),
                map50_95=float(map50_95),
            )
        )

    return tuple(summaries)


def normalize_confusion_matrix(confusion_matrix: np.ndarray) -> np.ndarray:
    """Normalize confusion matrix rows while handling empty rows safely."""
    row_sums = confusion_matrix.sum(axis=1, keepdims=True)
    return np.divide(
        confusion_matrix,
        row_sums,
        out=np.zeros_like(confusion_matrix, dtype=float),
        where=row_sums > 0,
    )


def write_confusion_matrix_csv(output_path: Path, class_labels: list[str], matrix: np.ndarray) -> None:
    """Write a confusion matrix to CSV for easier inspection and reporting."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["actual/predicted", *class_labels])
        for class_label, row in zip(class_labels, matrix.tolist()):
            writer.writerow([class_label, *[round(float(value), 6) for value in row]])


def write_evaluation_summary(summary: EvaluationRunSummary) -> None:
    """Persist a compact YAML summary next to the evaluation artifacts."""
    summary_path = summary.save_directory / "evaluation_summary.yaml"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as summary_file:
        yaml.safe_dump(stringify_paths(asdict(summary)), summary_file, sort_keys=False)


def stringify_paths(value: Any) -> Any:
    """Convert Path instances in nested structures into YAML-safe strings."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: stringify_paths(nested_value) for key, nested_value in value.items()}
    if isinstance(value, list):
        return [stringify_paths(item) for item in value]
    if isinstance(value, tuple):
        return [stringify_paths(item) for item in value]
    return value


def _configure_matplotlib_cache() -> None:
    """Point Matplotlib to a writable cache directory before validation plots are generated."""
    cache_directory = Path(tempfile.gettempdir()) / "yolo_edge_matplotlib"
    cache_directory.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_directory))


def _validate_evaluation_arguments(split: str, image_size: int, batch_size: int, workers: int) -> None:
    """Validate evaluation inputs early so failures are explicit."""
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be one of train, val, or test.")
    if image_size < 32:
        raise ValueError("image_size must be at least 32.")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if workers < 0:
        raise ValueError("workers cannot be negative.")
