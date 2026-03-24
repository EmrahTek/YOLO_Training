"""Media input and output helpers for edge-friendly YOLO inference."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

try:
    import cv2
    import numpy as np
except ModuleNotFoundError as error:
    raise ModuleNotFoundError(
        "Missing runtime dependency while importing the video streamer module. "
        "Activate the virtual environment and run 'pip install -r requirements.txt'."
    ) from error


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class FramePacket:
    """Represent a single frame and its metadata."""

    frame: np.ndarray
    frame_index: int
    source_name: str


class VideoStreamer:
    """Provide a lightweight interface around OpenCV source handling."""

    def load_image(self, image_path: Path) -> FramePacket:
        """Load a single image from disk."""
        image_path = image_path.resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        frame = cv2.imread(str(image_path))
        if frame is None:
            raise ValueError(f"Failed to read image: {image_path}")

        return FramePacket(frame=frame, frame_index=0, source_name=image_path.name)

    def open_video_capture(self, source: int | str | Path) -> cv2.VideoCapture:
        """Open a video file, camera index, or RTSP stream."""
        normalized_source = self.normalize_source(source)
        capture = cv2.VideoCapture(normalized_source)
        if not capture.isOpened():
            raise ValueError(f"Unable to open media source: {normalized_source}")

        LOGGER.info("Opened media source: %s", normalized_source)
        return capture

    def read_stream_frame(
        self,
        capture: cv2.VideoCapture,
        frame_index: int,
        source_name: str,
    ) -> FramePacket | None:
        """Read the next available frame from a stream."""
        success, frame = capture.read()
        if not success or frame is None:
            return None

        return FramePacket(frame=frame, frame_index=frame_index, source_name=source_name)

    def create_video_writer(
        self,
        output_path: Path,
        frame_width: int,
        frame_height: int,
        frames_per_second: float,
    ) -> cv2.VideoWriter:
        """Create an output writer for annotated videos."""
        output_path = output_path.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            frames_per_second,
            (frame_width, frame_height),
        )
        if not writer.isOpened():
            raise ValueError(f"Unable to create video writer for: {output_path}")

        return writer

    def save_image(self, output_path: Path, frame: np.ndarray) -> None:
        """Save an annotated image to disk."""
        output_path = output_path.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), frame):
            raise ValueError(f"Failed to write image to: {output_path}")

    def release_capture(self, capture: cv2.VideoCapture) -> None:
        """Release a capture object when it is open."""
        if capture.isOpened():
            capture.release()

    def display_frame(
        self,
        window_name: str,
        frame: np.ndarray,
        max_width: int | None = None,
        max_height: int | None = None,
    ) -> None:
        """Display a frame for interactive inspection with optional scaling."""
        display_frame = self.resize_for_display(
            frame=frame,
            max_width=max_width,
            max_height=max_height,
        )
        cv2.imshow(window_name, display_frame)

    def should_close_window(self, delay_milliseconds: int = 1) -> bool:
        """Return True when the user presses a close shortcut."""
        pressed_key = cv2.waitKey(delay_milliseconds) & 0xFF
        return pressed_key in {27, ord("q")}

    def destroy_windows(self) -> None:
        """Close all OpenCV windows."""
        cv2.destroyAllWindows()

    def resize_for_display(
        self,
        frame: np.ndarray,
        max_width: int | None = None,
        max_height: int | None = None,
    ) -> np.ndarray:
        """Resize a frame for display while preserving aspect ratio."""
        if max_width is None and max_height is None:
            return frame

        frame_height, frame_width = frame.shape[:2]
        width_scale = (max_width / frame_width) if max_width is not None else 1.0
        height_scale = (max_height / frame_height) if max_height is not None else 1.0
        scale = min(width_scale, height_scale, 1.0)

        if scale >= 1.0:
            return frame

        resized_width = max(1, int(frame_width * scale))
        resized_height = max(1, int(frame_height * scale))
        return cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)

    def normalize_source(self, source: int | str | Path) -> int | str:
        """Normalize CLI source values for OpenCV."""
        if isinstance(source, Path):
            return str(source.resolve())

        if isinstance(source, str):
            stripped_source = source.strip()
            return int(stripped_source) if stripped_source.isdigit() else stripped_source

        return source
