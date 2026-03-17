"""
Purpose:
    Centralized custom exception classes for cleaner error handling.

Why this module exists:
    Clean code is easier to maintain when domain-specific failures are
    described with explicit exception types.

Resources:
    - Python Exceptions: https://docs.python.org/3/tutorial/errors.html
"""

class ProjectError(Exception):
    """Base exception for all project-specific failures."""


class ConfigError(ProjectError):
    """Raised when configuration loading or validation fails."""


class DatasetError(ProjectError):
    """Raised when dataset files are missing or inconsistent."""


class FileOperationError(ProjectError):
    """Raised when file copy, move, or write operations fail."""


class PredictionError(ProjectError):
    """Raised when inference or export operations fail."""
