"""Install simple launcher commands into the active project's virtual environment."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENV_BIN_DIRECTORY = PROJECT_ROOT / ".venv" / "bin"
PYTHON_PATH = VENV_BIN_DIRECTORY / "python3"
MAIN_PATH = PROJECT_ROOT / "main.py"

COMMAND_TO_MODE = {
    "image": "image",
    "video": "video",
    "webcam": "webcam",
    "external-camera": "external-camera",
    "external_camera": "external-camera",
}


def build_launcher_script(mode: str) -> str:
    """Build the shell script content for one launcher command."""
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            'set -euo pipefail',
            f'exec "{PYTHON_PATH}" "{MAIN_PATH}" "{mode}" "$@"',
            "",
        ]
    )


def install_shortcuts() -> None:
    """Create executable launcher scripts in the virtual environment bin directory."""
    if not VENV_BIN_DIRECTORY.exists():
        raise FileNotFoundError(f"Virtual environment bin directory not found: {VENV_BIN_DIRECTORY}")

    if not PYTHON_PATH.exists():
        raise FileNotFoundError(f"Virtual environment interpreter not found: {PYTHON_PATH}")

    for command_name, mode in COMMAND_TO_MODE.items():
        launcher_path = VENV_BIN_DIRECTORY / command_name
        launcher_path.write_text(build_launcher_script(mode), encoding="utf-8")
        launcher_path.chmod(0o755)
        print(f"Installed launcher: {launcher_path}")


if __name__ == "__main__":
    install_shortcuts()
