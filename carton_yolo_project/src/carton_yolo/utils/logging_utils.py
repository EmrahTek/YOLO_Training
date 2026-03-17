"""
Purpose:
    Logging setup for console and file output.

Why this module exists:
    Computer vision workflows are much easier to debug when all stages
    write consistent logs to both terminal and file.

Resources:
    - Python logging cookbook:
      https://docs.python.org/3/howto/logging-cookbook.html
"""

from __future__ import annotations

import logging
import logging.config
from pathlib import Path

import yaml

from carton_yolo.constants import DEFAULT_LOG_FILE_NAME
from carton_yolo.utils.paths import ensure_directory, get_project_root


def setup_logging(logging_config_path: Path | None = None) -> None:
    """Configure application logging from YAML or fall back to basic config."""
    project_root = get_project_root()
    logs_dir = ensure_directory(project_root / "logs")
    default_log_path = logs_dir / DEFAULT_LOG_FILE_NAME

    if logging_config_path is not None and logging_config_path.exists():
        with logging_config_path.open("r", encoding="utf-8") as file_handle:
            config = yaml.safe_load(file_handle)

        handlers = config.get("handlers", {})
        file_handler = handlers.get("file")
        if isinstance(file_handler, dict):
            file_handler["filename"] = str(default_log_path)

        logging.config.dictConfig(config)
        return

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(default_log_path, encoding="utf-8"),
        ],
    )
