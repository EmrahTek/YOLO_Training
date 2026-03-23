"""Compatibility re-export for the dataset manager module."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_DIRECTORY = PROJECT_ROOT / "src"
if str(SOURCE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIRECTORY))

from yolo_edge_pipeline.data.dataset_manager import DatasetDescription
from yolo_edge_pipeline.data.dataset_manager import DatasetIssue
from yolo_edge_pipeline.data.dataset_manager import DatasetManager
from yolo_edge_pipeline.data.dataset_manager import DatasetValidationError


__all__ = [
    "DatasetDescription",
    "DatasetIssue",
    "DatasetManager",
    "DatasetValidationError",
]
