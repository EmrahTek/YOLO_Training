"""
Purpose:
    Main command-line entrypoint for the carton YOLO workflow.

Why this module exists:
    The project is intentionally modular. This file only parses commands,
    loads configuration, sets up logging, and dispatches work to the
    appropriate module.

Supported commands:
    - prepare_dataset
    - train
    - validate
    - image
    - video
    - camera
    - export_model

Resources:
    - argparse docs: https://docs.python.org/3/library/argparse.html
    - Ultralytics docs: https://docs.ultralytics.com/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from carton_yolo.config import load_app_config
from carton_yolo.dataset import prepare_dataset
from carton_yolo.exceptions import ProjectError
from carton_yolo.model_export import export_model
from carton_yolo.predictor import predict_camera, predict_images, predict_video
from carton_yolo.trainer import train_model
from carton_yolo.utils.logging_utils import setup_logging
from carton_yolo.validator import validate_model

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Carton YOLO pipeline for dataset preparation, training, validation, and inference."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to the project configuration YAML file.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("prepare_dataset", help="Prepare train/val dataset from CVAT export.")
    subparsers.add_parser("train", help="Train the YOLO model.")
    subparsers.add_parser("validate", help="Validate the trained model.")

    image_parser = subparsers.add_parser("image", help="Run inference on image directory or image file.")
    image_parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Path to an image or directory of images. Defaults to data/demo/unseen_images.",
    )

    video_parser = subparsers.add_parser("video", help="Run inference on a video file.")
    video_parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Path to the video file.",
    )

    camera_parser = subparsers.add_parser("camera", help="Run real-time webcam inference.")
    camera_parser.add_argument(
        "--source",
        type=int,
        default=None,
        help="Webcam index. Defaults to the value from config.",
    )

    subparsers.add_parser("export_model", help="Export the trained model.")

    return parser


def main() -> int:
    """Run the CLI application and return a process exit code."""
    parser = build_parser()
    args = parser.parse_args()

    config_path = args.config.resolve()
    setup_logging(config_path.parent / "logging.yaml")

    try:
        config = load_app_config(config_path)

        if args.command == "prepare_dataset":
            summary = prepare_dataset(config)
            LOGGER.info(
                "Prepared dataset successfully. Total=%s, Train=%s, Val=%s",
                summary.total_images,
                summary.train_images,
                summary.val_images,
            )
        elif args.command == "train":
            train_model(config)
        elif args.command == "validate":
            validate_model(config)
        elif args.command == "image":
            predict_images(config, source=args.source)
        elif args.command == "video":
            predict_video(config, source=args.source)
        elif args.command == "camera":
            predict_camera(config, camera_index=args.source)
        elif args.command == "export_model":
            export_model(config)
        else:
            parser.error(f"Unsupported command: {args.command}")
            return 2
    except ProjectError as exc:
        LOGGER.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        LOGGER.warning("Execution interrupted by user.")
        return 130
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Unexpected failure: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
