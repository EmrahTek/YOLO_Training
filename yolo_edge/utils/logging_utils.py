"""Centralized logging utilities for the project."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def configure_logging(log_level: str = "INFO", log_directory: Path | None = None) -> None:
    """Configure console and optional file logging for the application."""
    logging_level = getattr(logging, log_level.upper(), logging.INFO)
    handlers: list[logging.Handler] = [logging.StreamHandler()]

    if log_directory is not None:
        log_directory.mkdir(parents=True, exist_ok=True)
        log_file_path = log_directory / "application.log"
        handlers.append(
            RotatingFileHandler(
                filename=log_file_path,
                maxBytes=1_000_000,
                backupCount=3,
            )
        )

    logging.basicConfig(
        level=logging_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )
