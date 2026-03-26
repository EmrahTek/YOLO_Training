"""Tests for CLI argument validation."""

from __future__ import annotations

import unittest

from yolo_edge.cli import build_argument_parser
from yolo_edge.cli import parse_arguments
from yolo_edge.cli import validate_arguments


class CliValidationTestCase(unittest.TestCase):
    """Validate source-specific CLI argument rules."""

    def test_image_source_requires_path(self) -> None:
        """Image mode should parse correctly with only the mode name."""
        parser = build_argument_parser()
        arguments = parser.parse_args(["predict", "image"])
        validate_arguments(arguments, parser)

    def test_webcam_source_rejects_path(self) -> None:
        """Webcam inference should not accept a file path argument."""
        parser = build_argument_parser()
        arguments = parser.parse_args(["predict", "webcam", "--path", "sample.jpg"])

        with self.assertRaises(SystemExit):
            validate_arguments(arguments, parser)

    def test_video_source_accepts_path(self) -> None:
        """Video inference should accept a valid path argument."""
        parser = build_argument_parser()
        arguments = parser.parse_args(["predict", "video", "--path", "sample.mp4"])
        validate_arguments(arguments, parser)

    def test_predict_confidence_below_threshold_is_rejected(self) -> None:
        """Prediction confidence lower than 0.50 should be rejected."""
        parser = build_argument_parser()
        arguments = parser.parse_args(["predict", "image", "--confidence", "0.49"])

        with self.assertRaises(SystemExit):
            validate_arguments(arguments, parser)

    def test_external_camera_mode_is_supported(self) -> None:
        """External camera should be a valid simplified mode."""
        parser = build_argument_parser()
        arguments = parser.parse_args(["predict", "external-camera"])
        validate_arguments(arguments, parser)

    def test_simple_image_launcher_arguments_expand_to_predict_command(self) -> None:
        """The one-word launcher syntax should map to the predict subcommand."""
        arguments = parse_arguments(["image"])

        self.assertEqual(arguments.command, "predict")
        self.assertEqual(arguments.source, "image")
        self.assertEqual(arguments.confidence, 0.5)
        self.assertIsNotNone(arguments.device)

    def test_train_subcommand_is_available(self) -> None:
        """The project should expose a dedicated train subcommand."""
        arguments = parse_arguments(["train"])

        self.assertEqual(arguments.command, "train")
        self.assertIsNotNone(arguments.device)

    def test_evaluate_subcommand_is_available(self) -> None:
        """The project should expose a dedicated evaluate subcommand."""
        arguments = parse_arguments(["evaluate"])

        self.assertEqual(arguments.command, "evaluate")
        self.assertIsNotNone(arguments.device)


if __name__ == "__main__":
    unittest.main()
