"""Automatic Git commit and push watcher."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import subprocess
import time
from threading import Lock

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from yolo_edge_pipeline.utils.logging_utils import configure_logging


LOGGER = logging.getLogger(__name__)
DEFAULT_IGNORED_DIRECTORIES = {".git", "__pycache__", ".venv", "venv", "build", "dist", ".pytest_cache"}
DEFAULT_IGNORED_SUFFIXES = {".pyc", ".pyo", ".tmp", ".swp", ".lock"}


class GitAutomationError(RuntimeError):
    """Raised when an automated Git command fails."""


class GitChangeHandler(FileSystemEventHandler):
    """Trigger Git synchronization for changed files."""

    def __init__(self, repository_root: Path, debounce_seconds: float) -> None:
        """Store watcher configuration."""
        self._repository_root = repository_root.resolve()
        self._debounce_seconds = debounce_seconds
        self._last_processed_times: dict[Path, float] = {}
        self._lock = Lock()

    def on_any_event(self, event: FileSystemEvent) -> None:
        """React to non-directory file changes."""
        if event.is_directory:
            return

        source_path = Path(event.src_path).resolve()
        if self._should_ignore(source_path):
            return

        with self._lock:
            if not self._should_process(source_path):
                return
            self._last_processed_times[source_path] = time.monotonic()

        self._sync_file(source_path)

    def _should_ignore(self, path: Path) -> bool:
        """Ignore noisy or operational files."""
        relative_parts = path.relative_to(self._repository_root).parts
        if any(part in DEFAULT_IGNORED_DIRECTORIES for part in relative_parts):
            return True

        return path.suffix.lower() in DEFAULT_IGNORED_SUFFIXES

    def _should_process(self, path: Path) -> bool:
        """Debounce repeated writes for the same file."""
        last_processed_time = self._last_processed_times.get(path)
        if last_processed_time is None:
            return True

        return (time.monotonic() - last_processed_time) >= self._debounce_seconds

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
    parser.add_argument("--debounce-seconds", type=float, default=1.0)
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    return parser.parse_args()


def validate_repository_root(path: Path) -> Path:
    """Ensure the given directory is a Git repository."""
    repository_root = path.resolve()
    if not (repository_root / ".git").exists():
        raise FileNotFoundError(f"Git repository not found at: {repository_root}")

    return repository_root


def run_watcher() -> None:
    """Start the file watcher until interrupted."""
    arguments = parse_arguments()
    configure_logging(log_level=arguments.log_level, log_directory=arguments.log_dir)
    repository_root = validate_repository_root(arguments.watch_dir)

    observer = Observer()
    observer.schedule(
        GitChangeHandler(repository_root=repository_root, debounce_seconds=arguments.debounce_seconds),
        str(repository_root),
        recursive=True,
    )
    observer.start()
    LOGGER.info("Watching repository: %s", repository_root)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        LOGGER.info("Stopping Git automation watcher.")
        observer.stop()

    observer.join()


if __name__ == "__main__":
    run_watcher()
