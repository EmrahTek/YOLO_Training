"""Compatibility re-export for the detector module."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_DIRECTORY = PROJECT_ROOT / "src"
if str(SOURCE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIRECTORY))

from yolo_edge_pipeline.core.detector import Detection
from yolo_edge_pipeline.core.detector import InferenceResult
from yolo_edge_pipeline.core.detector import ObjectDetector


__all__ = ["Detection", "InferenceResult", "ObjectDetector"]
