"""
Purpose:
    File-system helpers for image discovery, copying, and validation.

Why this module exists:
    Dataset preparation tends to become messy when file operations are spread
    across multiple modules. This file keeps I/O behavior small and testable.

Resources:
    - pathlib documentation: https://docs.python.org/3/library/pathlib.html
    - shutil documentation: https://docs.python.org/3/library/shutil.html
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable

from carton_yolo.exceptions import FileOperationError


def list_files_with_extensions(directory: Path, extensions: Iterable[str]) -> list[Path]:
    """Return sorted files in a directory matching the given extensions."""
    normalized = {ext.lower() for ext in extensions}
    if not directory.exists():
        return []
    return sorted(
        file_path
        for file_path in directory.iterdir()
        if file_path.is_file() and file_path.suffix.lower() in normalized
    )


def copy_file_safe(source: Path, target: Path) -> None:
    """Copy a file and raise a project-specific error if it fails."""
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    except OSError as exc:
        raise FileOperationError(
            f"Failed to copy file from '{source}' to '{target}'."
        ) from exc


def clear_directory(directory: Path) -> None:
    """Remove all files and subdirectories inside a directory."""
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)
        return

    for item in directory.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()
