"""CLI orchestration for the YOLO edge object detection application."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import TYPE_CHECKING

import yaml

from yolo_edge.config import DEFAULT_CONFIG_PATH
from yolo_edge.config import AppConfig
from yolo_edge.config import load_app_config
from yolo_edge.data.dataset_manager import DatasetManager
from yolo_edge.edge_export import EdgeExporter
from yolo_edge.evaluation import get_default_device as get_default_evaluation_device
from yolo_edge.evaluation import run_evaluation
from yolo_edge.training import get_default_device as get_default_training_device
from yolo_edge.training import run_training
from yolo_edge.utils.logging_utils import configure_logging

if TYPE_CHECKING:
    from yolo_edge.core.detector import DetectionSummary
    from yolo_edge.core.detector import ObjectDetector
    from yolo_edge.core.video_streamer import FramePacket
    from yolo_edge.core.video_streamer import VideoStreamer


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEGACY_SOURCE_ALIASES = {"external_camera": "external-camera"}
SIMPLE_SOURCES = {"image", "video", "webcam", "external-camera", "external_camera"}


def build_argument_parser(app_config: AppConfig | None = None) -> argparse.ArgumentParser:
    """Create the root CLI parser so it can be reused in tests and scripts."""
    resolved_config = app_config or load_app_config(DEFAULT_CONFIG_PATH)
    parser = argparse.ArgumentParser(
        description="Run YOLO object detection, dataset preparation, training, and edge exports."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to the YAML config file.")
    subparsers = parser.add_subparsers(dest="command")

    predict_parser = subparsers.add_parser("predict", help="Run inference on image, video, or camera sources.")
    _add_predict_arguments(predict_parser, resolved_config)

    inspect_parser = subparsers.add_parser("inspect-dataset", help="Inspect the extracted CVAT dataset.")
    _add_dataset_arguments(inspect_parser, resolved_config)

    prepare_parser = subparsers.add_parser("prepare-dataset", help="Prepare a clean training dataset.")
    _add_dataset_arguments(prepare_parser, resolved_config)
    prepare_parser.add_argument("--overwrite", action="store_true")

    train_parser = subparsers.add_parser("train", help="Prepare the dataset and train a custom YOLO model.")
    _add_train_arguments(train_parser, resolved_config)

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate a trained model on the prepared dataset.")
    _add_evaluate_arguments(evaluate_parser, resolved_config)

    export_parser = subparsers.add_parser("export", help="Export trained weights for Raspberry Pi deployment.")
    _add_export_arguments(export_parser, resolved_config)

    return parser


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse arguments while supporting the existing one-word launcher mode."""
    raw_arguments = list(sys.argv[1:] if argv is None else argv)
    if raw_arguments:
        first_token = LEGACY_SOURCE_ALIASES.get(raw_arguments[0], raw_arguments[0])
        if first_token in SIMPLE_SOURCES:
            raw_arguments = ["predict", first_token, *raw_arguments[1:]]

    config_path = _extract_config_path(raw_arguments)
    app_config = load_app_config(config_path)
    parser = build_argument_parser(app_config)
    arguments = parser.parse_args(raw_arguments)
    if arguments.command is None:
        parser.error("A command is required. Use predict, train, evaluate, inspect-dataset, prepare-dataset, or export.")
    validate_arguments(arguments, parser)
    return arguments


def validate_arguments(arguments: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Validate command-specific argument combinations."""
    if arguments.command == "predict":
        if arguments.source in {"webcam", "external-camera"} and arguments.path is not None:
            parser.error("--path cannot be used when source is webcam or external-camera.")
        if arguments.image_delay_ms < 1:
            parser.error("--image-delay-ms must be greater than 0.")
        if not 0.50 <= arguments.confidence <= 1.0:
            parser.error("--confidence must be between 0.50 and 1.0.")


def resolve_legacy_path(path: Path | None) -> Path | None:
    """Map legacy project paths to the current folder structure when possible."""
    if path is None:
        return None

    candidate_path = path
    if candidate_path.exists():
        return candidate_path

    legacy_rewrites = {
        Path("Data/images"): Path("data/images"),
        Path("Data/Caton_Hause"): Path("data/cvat_exports/caton_hause"),
        Path("yolov8n.pt"): Path("models/yolov8n.pt"),
        Path("yolo11n.pt"): Path("models/yolo11n.pt"),
        Path("yolo11n-obb.pt"): Path("models/yolo11n-obb.pt"),
        Path("yolov10n.pt"): Path("models/yolov10n.pt"),
        Path("yolov8m-cls.pt"): Path("models/yolov8m-cls.pt"),
        Path("yolov8n-pose.pt"): Path("models/yolov8n-pose.pt"),
        Path("yolov8n-seg.pt"): Path("models/yolov8n-seg.pt"),
    }

    normalized_candidate = Path(str(candidate_path).strip())
    for legacy_prefix, current_prefix in legacy_rewrites.items():
        try:
            relative_path = normalized_candidate.relative_to(legacy_prefix)
            rewritten_path = current_prefix / relative_path
            if rewritten_path.exists():
                LOGGER.warning("Resolved legacy path %s -> %s", candidate_path, rewritten_path)
                return rewritten_path
        except ValueError:
            continue

    return candidate_path


def run_image_inference(
    detector: "ObjectDetector",
    streamer: "VideoStreamer",
    image_path: Path,
    save_output: bool,
    output_dir: Path,
    show_output: bool,
    display_max_width: int,
    display_max_height: int,
    image_delay_milliseconds: int,
) -> None:
    """Run inference on one image or iterate through all images in a directory."""
    image_paths = resolve_image_paths(image_path)
    LOGGER.info("Image mode started with %s image(s).", len(image_paths))

    try:
        for current_image_path in image_paths:
            frame_packet = streamer.load_image(current_image_path)
            inference_result = detector.predict(frame_packet.frame)
            log_detection_summary(frame_packet, inference_result.summary)

            if save_output:
                output_path = output_dir / f"annotated_{current_image_path.name}"
                streamer.save_image(output_path, inference_result.annotated_frame)
                LOGGER.info("Saved annotated image to %s", output_path)

            if show_output:
                streamer.display_frame(
                    "YOLO Detection",
                    inference_result.annotated_frame,
                    max_width=display_max_width,
                    max_height=display_max_height,
                )
                if streamer.should_close_window(delay_milliseconds=image_delay_milliseconds):
                    LOGGER.info("User requested to stop image playback.")
                    break
    finally:
        if show_output:
            streamer.destroy_windows()


def run_stream_inference(
    detector: "ObjectDetector",
    streamer: "VideoStreamer",
    source: int | str | Path,
    source_name: str,
    save_output: bool,
    output_dir: Path,
    show_output: bool,
    display_max_width: int,
    display_max_height: int,
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
                LOGGER.info("End of stream reached for source=%s", source_name)
                break

            inference_result = detector.predict(frame_packet.frame)
            log_detection_summary(frame_packet, inference_result.summary)

            if writer is not None:
                writer.write(inference_result.annotated_frame)

            if show_output:
                streamer.display_frame(
                    "YOLO Detection",
                    inference_result.annotated_frame,
                    max_width=display_max_width,
                    max_height=display_max_height,
                )
                if streamer.should_close_window():
                    LOGGER.info("User requested to close the visualization window.")
                    break

            frame_index += 1
    finally:
        if writer is not None:
            writer.release()
        streamer.release_capture(capture)
        if show_output:
            streamer.destroy_windows()


def log_detection_summary(frame_packet: "FramePacket", summary: "DetectionSummary") -> None:
    """Write a detailed inference summary to the logger."""
    if summary.has_detections:
        LOGGER.info(
            "source=%s frame=%s detected=yes total=%s classes=%s average_confidence=%.4f",
            frame_packet.source_name,
            frame_packet.frame_index,
            summary.total_detections,
            summary.class_counts,
            summary.average_confidence,
        )
        return

    LOGGER.info(
        "source=%s frame=%s detected=no total=0 classes={} average_confidence=0.0000",
        frame_packet.source_name,
        frame_packet.frame_index,
    )


def sanitize_source_name(source_name: str) -> str:
    """Create a file-system-safe identifier from a source name."""
    return source_name.replace("://", "_").replace("/", "_").replace(":", "_").replace(" ", "_")


def resolve_image_paths(path_argument: Path | None) -> list[Path]:
    """Resolve image mode inputs into an ordered image list."""
    candidate_path = resolve_legacy_path(path_argument)
    if candidate_path is None:
        raise FileNotFoundError("Unable to resolve the image source path.")

    candidate_path = candidate_path.resolve()
    if candidate_path.is_file():
        return [candidate_path]

    if candidate_path.is_dir():
        image_paths = sorted(
            path for path in candidate_path.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        )
        if not image_paths:
            raise FileNotFoundError(f"No image files were found in directory: {candidate_path}")
        return image_paths

    raise FileNotFoundError(f"Image path not found: {candidate_path}")


def resolve_video_path(path_argument: Path | None) -> Path:
    """Resolve a video path from an explicit file or a default data directory."""
    candidate_path = resolve_legacy_path(path_argument)
    if candidate_path is None:
        raise FileNotFoundError("Unable to resolve the video source path.")

    candidate_path = candidate_path.resolve()
    if candidate_path.is_file():
        return candidate_path

    if candidate_path.is_dir():
        video_paths = sorted(
            path for path in candidate_path.iterdir()
            if path.is_file() and path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}
        )
        if len(video_paths) == 1:
            return video_paths[0]
        if not video_paths:
            raise FileNotFoundError(
                f"No video files were found in directory: {candidate_path}. "
                "Place a video in data/videos or pass --path."
            )
        raise ValueError(
            f"Multiple video files were found in {candidate_path}. Please pass --path to choose one explicitly."
        )

    raise FileNotFoundError(f"Video path not found: {candidate_path}")


def warn_if_generic_model_for_custom_dataset(model_path: Path, dataset_root: Path) -> None:
    """Log a warning when a generic pretrained model is used with a custom carton dataset project."""
    generic_model_names = {
        "yolov8n.pt",
        "yolov8s.pt",
        "yolov8m.pt",
        "yolov8l.pt",
        "yolov8x.pt",
        "yolo11n.pt",
        "yolo11s.pt",
        "yolo11m.pt",
        "yolo11l.pt",
        "yolo11x.pt",
    }
    if model_path.name not in generic_model_names:
        return

    dataset_yaml_path = dataset_root / "data.yaml"
    if not dataset_yaml_path.exists():
        return

    with dataset_yaml_path.open("r", encoding="utf-8") as yaml_file:
        yaml_content = yaml.safe_load(yaml_file) or {}

    dataset_names = yaml_content.get("names", {})
    custom_classes = [str(value) for _, value in sorted(dataset_names.items())]
    LOGGER.warning(
        "You are using a generic pretrained model (%s). It will detect default COCO classes, not your custom carton classes %s. "
        "To detect cartons correctly, you need a model trained on this dataset.",
        model_path.name,
        custom_classes,
    )


def main(argv: list[str] | None = None) -> None:
    """Run the object detection application and its related workflows."""
    configure_logging()

    try:
        from yolo_edge.core.detector import ObjectDetector
        from yolo_edge.core.video_streamer import VideoStreamer

        arguments = parse_arguments(argv)
        configure_logging(log_level=arguments.log_level, log_directory=arguments.log_dir)

        if arguments.command == "predict":
            _run_predict_command(arguments, ObjectDetector, VideoStreamer)
            return

        if arguments.command == "inspect-dataset":
            _run_inspect_dataset_command(arguments)
            return

        if arguments.command == "prepare-dataset":
            _run_prepare_dataset_command(arguments)
            return

        if arguments.command == "train":
            _run_train_command(arguments)
            return

        if arguments.command == "evaluate":
            _run_evaluate_command(arguments)
            return

        if arguments.command == "export":
            _run_export_command(arguments)
            return
    except ModuleNotFoundError as error:
        LOGGER.error("%s", error)
        LOGGER.error("Current interpreter: %s", sys.executable)
        LOGGER.error("Recommended interpreter: %s", PROJECT_ROOT / ".venv" / "bin" / "python3")
        LOGGER.error(
            "Run with the project interpreter, for example: %s main.py image --model-path runs/train/carton_detector_gpu/weights/best.pt",
            PROJECT_ROOT / ".venv" / "bin" / "python3",
        )
        LOGGER.error("If needed, recreate the virtual environment because your current activation script may point to an old path.")
        LOGGER.error("Install the required packages with: pip install -r requirements.txt")
        raise SystemExit(1) from error
    except (FileNotFoundError, ValueError) as error:
        LOGGER.error("%s", error)
        raise SystemExit(1) from error
    except KeyboardInterrupt:
        LOGGER.warning("Execution interrupted by user.")
        raise SystemExit(130)
    except Exception as error:
        LOGGER.exception("Unexpected runtime error: %s", error)
        raise SystemExit(1) from error


def _run_predict_command(arguments: argparse.Namespace, object_detector_type: type["ObjectDetector"], video_streamer_type: type["VideoStreamer"]) -> None:
    """Execute the prediction workflow."""
    arguments.path = resolve_legacy_path(arguments.path)
    arguments.model_path = resolve_legacy_path(arguments.model_path)
    if arguments.source == "external-camera" and arguments.camera_index == "0":
        arguments.camera_index = arguments.external_camera_index

    detector = object_detector_type(
        model_path=arguments.model_path,
        confidence_threshold=arguments.confidence,
        device=arguments.device,
        image_size=arguments.image_size,
    )
    warn_if_generic_model_for_custom_dataset(arguments.model_path, arguments.dataset_root)
    streamer = video_streamer_type()

    if arguments.source == "image":
        run_image_inference(
            detector=detector,
            streamer=streamer,
            image_path=arguments.path if arguments.path is not None else arguments.default_image_directory,
            save_output=arguments.save_output,
            output_dir=arguments.output_dir,
            show_output=not arguments.no_show,
            display_max_width=arguments.display_max_width,
            display_max_height=arguments.display_max_height,
            image_delay_milliseconds=arguments.image_delay_ms,
        )
        return

    if arguments.source == "video":
        video_path = resolve_video_path(arguments.path if arguments.path is not None else arguments.default_video_directory)
        run_stream_inference(
            detector=detector,
            streamer=streamer,
            source=video_path,
            source_name=video_path.name,
            save_output=arguments.save_output,
            output_dir=arguments.output_dir,
            show_output=not arguments.no_show,
            display_max_width=arguments.display_max_width,
            display_max_height=arguments.display_max_height,
        )
        return

    run_stream_inference(
        detector=detector,
        streamer=streamer,
        source=arguments.camera_index,
        source_name=str(arguments.camera_index),
        save_output=arguments.save_output,
        output_dir=arguments.output_dir,
        show_output=not arguments.no_show,
        display_max_width=arguments.display_max_width,
        display_max_height=arguments.display_max_height,
    )


def _run_inspect_dataset_command(arguments: argparse.Namespace) -> None:
    """Inspect the dataset and log the most important health checks."""
    dataset_manager = DatasetManager()
    description = dataset_manager.inspect_dataset_directory(arguments.dataset_root)
    LOGGER.info(
        "dataset=%s images=%s labels=%s labeled_images=%s missing_labels=%s invalid_labels=%s classes=%s",
        description.dataset_root,
        description.image_count,
        description.label_count,
        description.labeled_image_count,
        len(description.missing_labels),
        len(description.invalid_label_files),
        description.class_names,
    )
    if description.missing_labels:
        LOGGER.warning("Missing label stems: %s", list(description.missing_labels))
    if description.invalid_label_files:
        LOGGER.warning("Invalid label files: %s", list(description.invalid_label_files))


def _run_prepare_dataset_command(arguments: argparse.Namespace) -> None:
    """Create the cleaned training dataset and log the result."""
    dataset_manager = DatasetManager()
    prepared_dataset_directory = dataset_manager.create_training_dataset(
        dataset_root=arguments.dataset_root,
        image_source_directory=arguments.image_source_dir,
        output_directory=arguments.prepared_dataset_dir,
        validation_ratio=arguments.validation_ratio,
        overwrite=arguments.overwrite,
        random_seed=arguments.random_seed,
    )
    LOGGER.info("Prepared dataset available at %s", prepared_dataset_directory)


def _run_train_command(arguments: argparse.Namespace) -> None:
    """Delegate to the training workflow."""
    run_training(
        dataset_root=arguments.dataset_root,
        image_source_directory=arguments.image_source_dir,
        prepared_dataset_directory=arguments.prepared_dataset_dir,
        base_model=arguments.base_model,
        epochs=arguments.epochs,
        image_size=arguments.image_size,
        batch_size=arguments.batch_size,
        device=str(arguments.device if arguments.device is not None else get_default_training_device()),
        validation_ratio=arguments.validation_ratio,
        random_seed=arguments.random_seed,
        project_directory=arguments.project_dir,
        run_name=arguments.run_name,
        overwrite=arguments.overwrite,
        patience=arguments.patience,
        workers=arguments.workers,
        cache=arguments.cache,
        resume=arguments.resume,
    )


def _run_export_command(arguments: argparse.Namespace) -> None:
    """Export trained weights and optionally benchmark the native model."""
    exporter = EdgeExporter()
    source_model = resolve_legacy_path(arguments.source_model)
    if source_model is None:
        raise FileNotFoundError("Unable to resolve the export source model.")

    artifacts = exporter.export_model(
        model_path=source_model,
        export_directory=arguments.export_dir,
        formats=tuple(arguments.formats),
        image_size=arguments.image_size,
        device=arguments.device,
        half=arguments.half,
        int8=arguments.int8,
        dynamic=arguments.dynamic,
    )
    LOGGER.info("Exported %s artifact(s).", len(artifacts))

    if arguments.benchmark_image is not None:
        benchmark_summary = exporter.benchmark_native_model(
            model_path=source_model,
            image_path=arguments.benchmark_image,
            confidence=arguments.confidence,
            image_size=arguments.image_size,
            device=arguments.device,
            runs=arguments.benchmark_runs,
        )
        LOGGER.info("Benchmark summary: %s", benchmark_summary)


def _run_evaluate_command(arguments: argparse.Namespace) -> None:
    """Evaluate a trained model and persist the resulting metrics."""
    run_evaluation(
        model_path=resolve_legacy_path(arguments.model_path) or arguments.model_path,
        dataset_yaml_path=resolve_legacy_path(arguments.dataset_yaml) or arguments.dataset_yaml,
        split=arguments.split,
        image_size=arguments.image_size,
        batch_size=arguments.batch_size,
        device=str(arguments.device if arguments.device is not None else get_default_evaluation_device()),
        project_directory=arguments.project_dir,
        run_name=arguments.run_name,
        workers=arguments.workers,
    )


def _add_predict_arguments(parser: argparse.ArgumentParser, app_config: AppConfig) -> None:
    """Attach inference arguments to a parser."""
    parser.add_argument(
        "source",
        choices=("image", "video", "webcam", "external-camera"),
        help="Execution mode. Use image, video, webcam, or external-camera.",
    )
    parser.add_argument("--path", type=Path, help="Path to an image or video file.")
    parser.add_argument(
        "--camera-index",
        default=app_config.predict.default_camera_index,
        help="Camera index such as 0 or 1, or an RTSP URL.",
    )
    parser.add_argument("--model-path", type=Path, default=app_config.predict.model_path)
    parser.add_argument("--confidence", type=float, default=app_config.predict.confidence)
    parser.add_argument("--device", default=app_config.predict.device or get_default_training_device())
    parser.add_argument("--image-size", type=int, default=app_config.predict.image_size)
    parser.add_argument("--no-show", action="store_true", help="Disable the visualization window.")
    parser.add_argument("--save-output", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=app_config.predict.output_directory)
    parser.add_argument("--display-max-width", type=int, default=app_config.predict.display_max_width)
    parser.add_argument("--display-max-height", type=int, default=app_config.predict.display_max_height)
    parser.add_argument("--image-delay-ms", type=int, default=app_config.predict.image_delay_ms)
    parser.add_argument("--log-level", default=app_config.predict.log_level, choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    parser.add_argument("--log-dir", type=Path, default=app_config.predict.log_directory)
    parser.add_argument("--default-image-directory", type=Path, default=app_config.predict.image_directory)
    parser.add_argument("--default-video-directory", type=Path, default=app_config.predict.video_directory)
    parser.add_argument("--external-camera-index", default=app_config.predict.external_camera_index)
    parser.add_argument("--dataset-root", type=Path, default=app_config.dataset.dataset_root)


def _add_dataset_arguments(parser: argparse.ArgumentParser, app_config: AppConfig) -> None:
    """Attach dataset arguments to a parser."""
    parser.add_argument("--dataset-root", type=Path, default=app_config.dataset.dataset_root)
    parser.add_argument("--image-source-dir", type=Path, default=app_config.dataset.image_source_directory)
    parser.add_argument("--prepared-dataset-dir", type=Path, default=app_config.dataset.prepared_dataset_directory)
    parser.add_argument("--validation-ratio", type=float, default=app_config.dataset.validation_ratio)
    parser.add_argument("--random-seed", type=int, default=app_config.dataset.random_seed)
    parser.add_argument("--log-level", default=app_config.predict.log_level, choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    parser.add_argument("--log-dir", type=Path, default=app_config.predict.log_directory)


def _add_train_arguments(parser: argparse.ArgumentParser, app_config: AppConfig) -> None:
    """Attach training arguments to a parser."""
    _add_dataset_arguments(parser, app_config)
    parser.add_argument("--base-model", type=Path, default=app_config.train.base_model)
    parser.add_argument("--epochs", type=int, default=app_config.train.epochs)
    parser.add_argument("--image-size", type=int, default=app_config.train.image_size)
    parser.add_argument("--batch-size", type=int, default=app_config.train.batch_size)
    parser.add_argument("--device", default=app_config.train.device or get_default_training_device())
    parser.add_argument("--project-dir", type=Path, default=app_config.train.project_directory)
    parser.add_argument("--run-name", default=app_config.train.run_name)
    parser.add_argument("--patience", type=int, default=app_config.train.patience)
    parser.add_argument("--workers", type=int, default=app_config.train.workers)
    parser.add_argument("--cache", action="store_true", default=app_config.train.cache)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true", default=app_config.train.resume)


def _add_export_arguments(parser: argparse.ArgumentParser, app_config: AppConfig) -> None:
    """Attach export arguments to a parser."""
    parser.add_argument("--source-model", type=Path, default=app_config.export.source_model)
    parser.add_argument("--export-dir", type=Path, default=app_config.export.export_directory)
    parser.add_argument("--formats", nargs="+", default=list(app_config.export.formats))
    parser.add_argument("--image-size", type=int, default=app_config.export.image_size)
    parser.add_argument("--half", action="store_true", default=app_config.export.half)
    parser.add_argument("--int8", action="store_true", default=app_config.export.int8)
    parser.add_argument("--dynamic", action="store_true", default=app_config.export.dynamic)
    parser.add_argument("--device", default=app_config.export.device)
    parser.add_argument("--benchmark-image", type=Path, default=app_config.export.benchmark_image)
    parser.add_argument("--benchmark-runs", type=int, default=app_config.export.benchmark_runs)
    parser.add_argument("--confidence", type=float, default=app_config.predict.confidence)
    parser.add_argument("--log-level", default=app_config.predict.log_level, choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    parser.add_argument("--log-dir", type=Path, default=app_config.predict.log_directory)


def _add_evaluate_arguments(parser: argparse.ArgumentParser, app_config: AppConfig) -> None:
    """Attach evaluation arguments to a parser."""
    parser.add_argument("--model-path", type=Path, default=app_config.evaluation.model_path)
    parser.add_argument("--dataset-yaml", type=Path, default=app_config.evaluation.dataset_yaml)
    parser.add_argument("--split", choices=("train", "val", "test"), default=app_config.evaluation.split)
    parser.add_argument("--image-size", type=int, default=app_config.evaluation.image_size)
    parser.add_argument("--batch-size", type=int, default=app_config.evaluation.batch_size)
    parser.add_argument("--device", default=app_config.evaluation.device or get_default_evaluation_device())
    parser.add_argument("--project-dir", type=Path, default=app_config.evaluation.project_directory)
    parser.add_argument("--run-name", default=app_config.evaluation.run_name)
    parser.add_argument("--workers", type=int, default=app_config.evaluation.workers)
    parser.add_argument("--log-level", default=app_config.predict.log_level, choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    parser.add_argument("--log-dir", type=Path, default=app_config.predict.log_directory)


def _extract_config_path(raw_arguments: list[str]) -> Path | None:
    """Extract the config path from raw argv before the parser is built."""
    if "--config" not in raw_arguments:
        return None

    config_index = raw_arguments.index("--config")
    if config_index == len(raw_arguments) - 1:
        raise ValueError("--config must be followed by a path.")
    return Path(raw_arguments[config_index + 1])
