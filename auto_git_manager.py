"""Automatically commit and push file changes to Git."""

from __future__ import annotations

import argparse
import logging
import subprocess
import time
from pathlib import Path
from threading import Lock
from typing import Iterable

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer


LOGGER = logging.getLogger(__name__)
DEFAULT_IGNORED_DIRECTORIES = {".git", "__pycache__", ".venv", "venv", "build", "dist"}
DEFAULT_IGNORED_SUFFIXES = {".pyc", ".pyo", ".tmp", ".swp", ".lock"}


class GitAutomationError(RuntimeError):
    """Raised when an automated Git command fails."""


class GitChangeHandler(FileSystemEventHandler):
    """React to file system changes and trigger Git synchronization."""

    def __init__(
        self,
        repository_root: Path,
        debounce_seconds: float,
        ignored_directories: set[str],
        ignored_suffixes: set[str],
    ) -> None:
        """Initialize the file system handler."""
        self._repository_root = repository_root.resolve()
        self._debounce_seconds = debounce_seconds
        self._ignored_directories = ignored_directories
        self._ignored_suffixes = ignored_suffixes
        self._last_processed_times: dict[Path, float] = {}
        self._lock = Lock()

    def on_any_event(self, event: FileSystemEvent) -> None:
        """Handle file changes while intentionally ignoring noisy paths."""
        if event.is_directory:
            return

        source_path = Path(event.src_path).resolve()
        if self._should_ignore(source_path):
            return

        with self._lock:
            if not self._should_process(source_path):
                return

            self._last_processed_times[source_path] = time.monotonic()

        try:
            self._sync_file(source_path)
        except GitAutomationError as error:
            LOGGER.error("Git synchronization failed for %s: %s", source_path, error)

    def _should_ignore(self, path: Path) -> bool:
        """Return True when the file should not trigger a Git action."""
        relative_parts = path.relative_to(self._repository_root).parts
        if any(part in self._ignored_directories for part in relative_parts):
            return True

        return path.suffix.lower() in self._ignored_suffixes

    def _should_process(self, path: Path) -> bool:
        """Debounce rapid file updates to avoid duplicate commits."""
        last_processed_time = self._last_processed_times.get(path)
        if last_processed_time is None:
            return True

        return (time.monotonic() - last_processed_time) >= self._debounce_seconds

    def _sync_file(self, path: Path) -> None:
        """Commit and push a single changed file with a deterministic message."""
        relative_path = path.relative_to(self._repository_root)
        commit_message = f"feat: add/update {relative_path.as_posix()}"
        commands = (
            ["git", "add", relative_path.as_posix()],
            ["git", "commit", "-m", commit_message],
            ["git", "push"],
        )

        for command in commands:
            self._run_command(command)

        LOGGER.info("Synchronized %s", relative_path.as_posix())

    def _run_command(self, command: list[str]) -> None:
        """Run a Git command and raise a domain-specific error on failure."""
        completed_process = subprocess.run(
            command,
            cwd=self._repository_root,
            check=False,
            capture_output=True,
            text=True,
        )

        if completed_process.returncode == 0:
            return

        # Git returns a non-zero code when there is nothing new to commit.
        stderr_output = completed_process.stderr.strip()
        stdout_output = completed_process.stdout.strip()
        combined_output = stderr_output or stdout_output

        if "nothing to commit" in combined_output.lower():
            LOGGER.info("No Git changes detected for command: %s", " ".join(command))
            return

        raise GitAutomationError(combined_output)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for the Git automation watcher."""
    parser = argparse.ArgumentParser(
        description="Watch a repository directory and automatically commit and push file changes."
    )
    parser.add_argument(
        "--watch-dir",
        type=Path,
        default=Path("."),
        help="Directory to monitor recursively. Defaults to the current directory.",
    )
    parser.add_argument(
        "--debounce-seconds",
        type=float,
        default=1.0,
        help="Minimum delay between repeated actions for the same file.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity level.",
    )
    return parser.parse_args()


def configure_logging(log_level: str) -> None:
    """Configure application logging with a compact and readable format."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def validate_repository_root(path: Path) -> Path:
    """Ensure the watched directory is a valid Git repository root."""
    repository_root = path.resolve()
    git_directory = repository_root / ".git"
    if not git_directory.exists():
        raise FileNotFoundError(f"Git repository not found at: {repository_root}")

    return repository_root


def start_observer(
    repository_root: Path,
    debounce_seconds: float,
    ignored_directories: Iterable[str],
    ignored_suffixes: Iterable[str],
) -> None:
    """Start the file system observer and keep it alive until interrupted."""
    event_handler = GitChangeHandler(
        repository_root=repository_root,
        debounce_seconds=debounce_seconds,
        ignored_directories=set(ignored_directories),
        ignored_suffixes=set(ignored_suffixes),
    )
    observer = Observer()
    observer.schedule(event_handler, str(repository_root), recursive=True)
    observer.start()

    LOGGER.info("Watching repository: %s", repository_root)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        LOGGER.info("Stopping Git automation watcher.")
        observer.stop()

    observer.join()


def main() -> None:
    """Run the Git automation watcher application."""
    arguments = parse_arguments()
    configure_logging(arguments.log_level)
    repository_root = validate_repository_root(arguments.watch_dir)
    start_observer(
        repository_root=repository_root,
        debounce_seconds=arguments.debounce_seconds,
        ignored_directories=DEFAULT_IGNORED_DIRECTORIES,
        ignored_suffixes=DEFAULT_IGNORED_SUFFIXES,
    )


if __name__ == "__main__":
    main()
