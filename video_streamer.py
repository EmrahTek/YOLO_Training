"""Media input utilities for image, video, and webcam sources."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class FramePacket:
    """Represent a single frame and its metadata."""

    frame: np.ndarray
    frame_index: int
    source_name: str


class VideoStreamer:
    """Provide a lightweight abstraction over OpenCV media sources."""

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
        """Open a video capture source from a file, device index, or stream URL."""
        resolved_source = self._normalize_source(source)
        capture = cv2.VideoCapture(resolved_source)

        if not capture.isOpened():
            raise ValueError(f"Unable to open media source: {resolved_source}")

        return capture

    def read_stream_frame(self, capture: cv2.VideoCapture, frame_index: int, source_name: str) -> FramePacket | None:
        """Read the next frame from an open capture stream."""
        success, frame = capture.read()
        if not success or frame is None:
            return None

        return FramePacket(frame=frame, frame_index=frame_index, source_name=source_name)

    def release_capture(self, capture: cv2.VideoCapture) -> None:
        """Release an OpenCV capture safely."""
        if capture.isOpened():
            capture.release()

    def display_frame(self, window_name: str, frame: np.ndarray) -> None:
        """Display a frame in a named OpenCV window."""
        # Keeping display logic here avoids coupling visualization to the detector.
        cv2.imshow(window_name, frame)

    def should_close_window(self, delay_milliseconds: int = 1) -> bool:
        """Return True when the user requests window closure using the keyboard."""
        pressed_key = cv2.waitKey(delay_milliseconds) & 0xFF
        return pressed_key in {27, ord("q")}

    def destroy_windows(self) -> None:
        """Close all OpenCV windows."""
        cv2.destroyAllWindows()

    def create_video_writer(
        self,
        output_path: Path,
        frame_width: int,
        frame_height: int,
        frames_per_second: float,
    ) -> cv2.VideoWriter:
        """Create a video writer for annotated stream output."""
        output_path = output_path.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # MP4 is broadly supported and keeps the artifact easy to inspect later.
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            frames_per_second,
            (frame_width, frame_height),
        )

        if not writer.isOpened():
            raise ValueError(f"Unable to create video writer for: {output_path}")

        return writer

    def save_image(self, output_path: Path, frame: np.ndarray) -> None:
        """Persist a single annotated image to disk."""
        output_path = output_path.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        success = cv2.imwrite(str(output_path), frame)
        if not success:
            raise ValueError(f"Failed to write image to: {output_path}")

    def _normalize_source(self, source: int | str | Path) -> int | str:
        """Normalize source values for OpenCV consumption."""
        if isinstance(source, Path):
            return str(source.resolve())

        if isinstance(source, str):
            stripped_source = source.strip()
            return int(stripped_source) if stripped_source.isdigit() else stripped_source

        return source
