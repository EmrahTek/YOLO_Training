"""Tests for CLI argument validation."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRECTORY = PROJECT_ROOT / "src"
if str(SOURCE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIRECTORY))

from yolo_edge_pipeline.cli import build_argument_parser
from yolo_edge_pipeline.cli import validate_arguments


class CliValidationTestCase(unittest.TestCase):
    """Validate source-specific CLI argument rules."""

    def test_image_source_requires_path(self) -> None:
        """Image inference must require a file path."""
        parser = build_argument_parser()
        arguments = parser.parse_args(["--source", "image"])

        with self.assertRaises(SystemExit):
            validate_arguments(arguments, parser)

    def test_webcam_source_rejects_path(self) -> None:
        """Webcam inference should not accept a file path argument."""
        parser = build_argument_parser()
        arguments = parser.parse_args(["--source", "webcam", "--path", "sample.jpg"])

        with self.assertRaises(SystemExit):
            validate_arguments(arguments, parser)

    def test_video_source_accepts_path(self) -> None:
        """Video inference should accept a valid path argument."""
        parser = build_argument_parser()
        arguments = parser.parse_args(["--source", "video", "--path", "sample.mp4"])
        validate_arguments(arguments, parser)


if __name__ == "__main__":
    unittest.main()
