"""Root entry point for the Git automation watcher."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_DIRECTORY = PROJECT_ROOT / "src"
if str(SOURCE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIRECTORY))

from yolo_edge_pipeline.tools.auto_git_manager import run_watcher


if __name__ == "__main__":
    run_watcher()
