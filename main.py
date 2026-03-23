"""CLI entry point for the YOLO object detection pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from detector import ObjectDetector
from video_streamer import FramePacket
from video_streamer import VideoStreamer


def parse_arguments() -> argparse.Namespace:
    """Parse CLI arguments and enforce source-specific requirements."""
    parser = argparse.ArgumentParser(
        description="Run YOLO object detection on an image, video, webcam, or RTSP stream."
    )
    parser.add_argument(
        "--source",
        required=True,
        choices=("image", "video", "webcam"),
        help="Input source type.",
    )
    parser.add_argument(
        "--path",
        type=Path,
        help="Path to the image or video file. Required for image and video sources.",
    )
    parser.add_argument(
        "--camera-index",
        default="0",
        help="Camera index for webcams or an RTSP URL for network streams. Defaults to 0.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("yolov8n.pt"),
        help="Path to the Ultralytics YOLO model checkpoint.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Confidence threshold used for object detection.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional inference device such as cpu or cuda:0.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display annotated output in an OpenCV window.",
    )
    parser.add_argument(
        "--save-output",
        action="store_true",
        help="Save annotated output to the output directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory used to store annotated output artifacts.",
    )
    arguments = parser.parse_args()
    validate_arguments(arguments)
    return arguments


def validate_arguments(arguments: argparse.Namespace) -> None:
    """Validate source-specific CLI argument combinations."""
    if arguments.source in {"image", "video"} and arguments.path is None:
        raise ValueError("--path is required when --source is image or video.")

    if arguments.source == "webcam" and arguments.camera_index is None:
        raise ValueError("--camera-index is required when --source is webcam.")


def run_image_inference(
    detector: ObjectDetector,
    streamer: VideoStreamer,
    image_path: Path,
    save_output: bool,
    output_dir: Path,
    show_output: bool,
) -> None:
    """Run inference for a single image."""
    frame_packet = streamer.load_image(image_path)
    inference_result = detector.predict(frame_packet.frame)

    print_detection_summary(frame_packet, inference_result.detections)

    if save_output:
        output_path = output_dir / f"annotated_{image_path.name}"
        streamer.save_image(output_path, inference_result.annotated_frame)
        print(f"Saved annotated image to: {output_path}")

    if show_output:
        streamer.display_frame("YOLO Detection", inference_result.annotated_frame)
        cv2.waitKey(0)
        streamer.destroy_windows()


def run_stream_inference(
    detector: ObjectDetector,
    streamer: VideoStreamer,
    source: int | str | Path,
    source_name: str,
    save_output: bool,
    output_dir: Path,
    show_output: bool,
) -> None:
    """Run inference for a video file, webcam device, or RTSP stream."""
    capture = streamer.open_video_capture(source)
    writer = None

    try:
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        frames_per_second = capture.get(cv2.CAP_PROP_FPS) or 30.0

        if save_output:
            output_path = output_dir / f"annotated_{sanitize_source_name(source_name)}.mp4"
            writer = streamer.create_video_writer(
                output_path=output_path,
                frame_width=frame_width,
                frame_height=frame_height,
                frames_per_second=frames_per_second,
            )
            print(f"Saving annotated stream to: {output_path}")

        frame_index = 0
        while True:
            frame_packet = streamer.read_stream_frame(capture, frame_index, source_name)
            if frame_packet is None:
                break

            inference_result = detector.predict(frame_packet.frame)
            print_detection_summary(frame_packet, inference_result.detections)

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


def print_detection_summary(frame_packet: FramePacket, detections: list) -> None:
    """Print a compact summary for each processed frame."""
    print(
        f"Processed {frame_packet.source_name} frame={frame_packet.frame_index} "
        f"detections={len(detections)}"
    )


def sanitize_source_name(source_name: str) -> str:
    """Create a file-system-safe source identifier."""
    sanitized = source_name.replace("://", "_").replace("/", "_").replace(" ", "_")
    return sanitized.replace(":", "_")


def main() -> None:
    """Run the CLI application."""
    arguments = parse_arguments()

    detector = ObjectDetector(
        model_path=arguments.model_path,
        confidence_threshold=arguments.confidence,
        device=arguments.device,
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
