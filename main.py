"""CLI entry point for the YOLO edge object detection application."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from logging_utils import configure_logging

if TYPE_CHECKING:
    from detector import ObjectDetector
    from video_streamer import FramePacket
    from video_streamer import VideoStreamer


LOGGER = logging.getLogger(__name__)


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the CLI parser so it can be reused in tests."""
    parser = argparse.ArgumentParser(
        description="Run YOLO object detection on images, videos, webcams, or RTSP streams."
    )
    parser.add_argument("--source", required=True, choices=("image", "video", "webcam"))
    parser.add_argument("--path", type=Path, help="Path to an image or video file.")
    parser.add_argument(
        "--camera-index",
        default="0",
        help="Camera index such as 0 or 1, or an RTSP URL.",
    )
    parser.add_argument("--model-path", type=Path, default=Path("yolov8n.pt"))
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--device", default=None)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--save-output", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    return parser


def parse_arguments() -> argparse.Namespace:
    """Parse and validate command line arguments."""
    parser = build_argument_parser()
    arguments = parser.parse_args()
    validate_arguments(arguments, parser)
    return arguments


def validate_arguments(arguments: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Validate source-specific argument combinations."""
    if arguments.source in {"image", "video"} and arguments.path is None:
        parser.error("--path is required when --source is image or video.")

    if arguments.source == "webcam" and arguments.path is not None:
        parser.error("--path cannot be used when --source is webcam.")


def run_image_inference(
    detector: "ObjectDetector",
    streamer: "VideoStreamer",
    image_path: Path,
    save_output: bool,
    output_dir: Path,
    show_output: bool,
) -> None:
    """Run inference on a single image."""
    frame_packet = streamer.load_image(image_path)
    inference_result = detector.predict(frame_packet.frame)
    log_detection_summary(frame_packet, inference_result.detections)

    if save_output:
        output_path = output_dir / f"annotated_{image_path.name}"
        streamer.save_image(output_path, inference_result.annotated_frame)
        LOGGER.info("Saved annotated image to %s", output_path)

    if show_output:
        streamer.display_frame("YOLO Detection", inference_result.annotated_frame)
        streamer.should_close_window(delay_milliseconds=0)
        streamer.destroy_windows()


def run_stream_inference(
    detector: "ObjectDetector",
    streamer: "VideoStreamer",
    source: int | str | Path,
    source_name: str,
    save_output: bool,
    output_dir: Path,
    show_output: bool,
) -> None:
    """Run frame-by-frame inference for a video file or live stream."""
    capture = streamer.open_video_capture(source)
    writer = None

    try:
        frame_width = int(capture.get(3)) or 640
        frame_height = int(capture.get(4)) or 480
        frames_per_second = capture.get(5) or 30.0

        if save_output:
            output_path = output_dir / f"annotated_{sanitize_source_name(source_name)}.mp4"
            writer = streamer.create_video_writer(
                output_path=output_path,
                frame_width=frame_width,
                frame_height=frame_height,
                frames_per_second=frames_per_second,
            )
            LOGGER.info("Saving annotated stream to %s", output_path)

        frame_index = 0
        while True:
            frame_packet = streamer.read_stream_frame(capture, frame_index, source_name)
            if frame_packet is None:
                break

            inference_result = detector.predict(frame_packet.frame)
            log_detection_summary(frame_packet, inference_result.detections)

            if writer is not None:
                writer.write(inference_result.annotated_frame)

            if show_output:
                streamer.display_frame("YOLO Detection", inference_result.annotated_frame)
                if streamer.should_close_window():
                    break

            frame_index += 1
    finally:
        if writer is not None:
            writer.release()
        streamer.release_capture(capture)
        if show_output:
            streamer.destroy_windows()


def log_detection_summary(frame_packet: "FramePacket", detections: tuple) -> None:
    """Write a compact inference summary to the logger."""
    LOGGER.info(
        "Processed source=%s frame=%s detections=%s",
        frame_packet.source_name,
        frame_packet.frame_index,
        len(detections),
    )


def sanitize_source_name(source_name: str) -> str:
    """Create a file-system-safe identifier from a source name."""
    return source_name.replace("://", "_").replace("/", "_").replace(":", "_").replace(" ", "_")


def main() -> None:
    """Run the object detection application."""
    from detector import ObjectDetector
    from video_streamer import VideoStreamer

    arguments = parse_arguments()
    configure_logging(log_level=arguments.log_level, log_directory=arguments.log_dir)

    detector = ObjectDetector(
        model_path=arguments.model_path,
        confidence_threshold=arguments.confidence,
        device=arguments.device,
        image_size=arguments.image_size,
    )
    streamer = VideoStreamer()

    if arguments.source == "image":
        run_image_inference(
            detector=detector,
            streamer=streamer,
            image_path=arguments.path,
            save_output=arguments.save_output,
            output_dir=arguments.output_dir,
            show_output=arguments.show,
        )
        return

    if arguments.source == "video":
        run_stream_inference(
            detector=detector,
            streamer=streamer,
            source=arguments.path,
            source_name=arguments.path.name,
            save_output=arguments.save_output,
            output_dir=arguments.output_dir,
            show_output=arguments.show,
        )
        return

    run_stream_inference(
        detector=detector,
        streamer=streamer,
        source=arguments.camera_index,
        source_name=str(arguments.camera_index),
        save_output=arguments.save_output,
        output_dir=arguments.output_dir,
        show_output=arguments.show,
    )


if __name__ == "__main__":
    main()
