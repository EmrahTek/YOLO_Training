"""Tests for CLI argument validation."""

from __future__ import annotations

import unittest

from yolo_edge.cli import build_argument_parser
from yolo_edge.cli import validate_arguments


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
