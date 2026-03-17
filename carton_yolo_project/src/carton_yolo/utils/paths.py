"""
Purpose:
    Path helper functions for consistent project-root based file handling.

Why this module exists:
    Relative paths become fragile very quickly in ML projects.
    This helper ensures that all important directories are resolved
    relative to the project root.

Resources:
    - pathlib documentation: https://docs.python.org/3/library/pathlib.html
"""

from __future__ import annotations

from pathlib import Path


def get_project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parents[3]


def resolve_from_root(relative_path: str | Path) -> Path:
    """Resolve a path relative to the project root."""
    root = get_project_root()
    return (root / Path(relative_path)).resolve()


def ensure_directory(path: Path) -> Path:
    """Create a directory if it does not exist and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path
