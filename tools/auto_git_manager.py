"""Polling-based Git automation watcher without external file-watch dependencies."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import subprocess
import time

from yolo_edge.utils.logging_utils import configure_logging


LOGGER = logging.getLogger(__name__)
DEFAULT_IGNORED_DIRECTORIES = {".git", "__pycache__", ".venv", "venv", "build", "dist", ".pytest_cache"}
DEFAULT_IGNORED_SUFFIXES = {".pyc", ".pyo", ".tmp", ".swp", ".lock"}


class GitAutomationError(RuntimeError):
    """Raised when an automated Git command fails."""


class PollingGitWatcher:
    """Watch a repository by periodically scanning file modification times."""

    def __init__(self, repository_root: Path, poll_interval_seconds: float) -> None:
        """Store watcher configuration."""
        self._repository_root = repository_root.resolve()
        self._poll_interval_seconds = poll_interval_seconds
        self._known_state: dict[Path, float] = {}

    def run(self) -> None:
        """Start the polling loop until interrupted."""
        self._known_state = self._snapshot_files()
        LOGGER.info("Watching repository: %s", self._repository_root)

        while True:
            time.sleep(self._poll_interval_seconds)
            current_state = self._snapshot_files()
            changed_files = sorted(path for path, mtime in current_state.items() if self._known_state.get(path) != mtime)
            self._known_state = current_state

            for changed_file in changed_files:
                self._sync_file(changed_file)

    def _snapshot_files(self) -> dict[Path, float]:
        """Collect the current file modification state of the repository."""
        state: dict[Path, float] = {}

        for directory_path, directory_names, file_names in os.walk(self._repository_root):
            directory = Path(directory_path)
            directory_names[:] = [name for name in directory_names if name not in DEFAULT_IGNORED_DIRECTORIES]

            for file_name in file_names:
                file_path = directory / file_name
                if file_path.suffix.lower() in DEFAULT_IGNORED_SUFFIXES:
                    continue
                state[file_path.resolve()] = file_path.stat().st_mtime

        return state

    def _sync_file(self, path: Path) -> None:
        """Run add, commit, and push for the changed file."""
        relative_path = path.relative_to(self._repository_root)
        commit_message = f"feat: add/update {relative_path.as_posix()}"

        for command in (
            ["git", "add", relative_path.as_posix()],
            ["git", "commit", "-m", commit_message],
            ["git", "push"],
        ):
            self._run_command(command)

        LOGGER.info("Synchronized %s", relative_path.as_posix())

    def _run_command(self, command: list[str]) -> None:
        """Run a Git command and tolerate empty commits gracefully."""
        completed_process = subprocess.run(
            command,
            cwd=self._repository_root,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed_process.returncode == 0:
            return

        combined_output = (completed_process.stderr or completed_process.stdout).strip()
        if "nothing to commit" in combined_output.lower():
            LOGGER.info("No changes detected for command: %s", " ".join(command))
            return

        raise GitAutomationError(combined_output)


def parse_arguments() -> argparse.Namespace:
    """Parse watcher CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Watch a repository and automatically git add, commit, and push changes."
    )
    parser.add_argument("--watch-dir", type=Path, default=Path("."))
    parser.add_argument("--poll-interval-seconds", type=float, default=1.0)
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    return parser.parse_args()


def validate_repository_root(path: Path) -> Path:
    """Ensure the given directory is a Git repository."""
    repository_root = path.resolve()
    if not (repository_root / ".git").exists():
        raise FileNotFoundError(f"Git repository not found at: {repository_root}")

    return repository_root


def main() -> None:
    """Run the polling watcher until interrupted."""
    arguments = parse_arguments()
    configure_logging(log_level=arguments.log_level, log_directory=arguments.log_dir)

    try:
        repository_root = validate_repository_root(arguments.watch_dir)
        PollingGitWatcher(
            repository_root=repository_root,
            poll_interval_seconds=arguments.poll_interval_seconds,
        ).run()
    except KeyboardInterrupt:
        LOGGER.info("Stopping Git automation watcher.")
    except Exception as error:
        LOGGER.exception("Git automation watcher failed: %s", error)
        raise SystemExit(1) from error
