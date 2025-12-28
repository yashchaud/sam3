"""Video stream handling for various sources."""

import cv2
import numpy as np
import threading
import time
from pathlib import Path
from typing import Iterator, Callable
from enum import Enum

from .config import FrameSource
from .frame_buffer import FrameBuffer, BufferedFrame


class StreamState(Enum):
    """State of the stream handler."""
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    FINISHED = "finished"


class StreamHandler:
    """
    Handles video stream input from various sources.

    Supports:
    - Video files (mp4, avi, etc.)
    - Webcam
    - RTSP streams
    - Image sequences
    """

    def __init__(
        self,
        source_type: FrameSource,
        source_path: str | None = None,
        webcam_id: int = 0,
        target_fps: float | None = None,
        buffer: FrameBuffer | None = None,
    ):
        self.source_type = source_type
        self.source_path = source_path
        self.webcam_id = webcam_id
        self.target_fps = target_fps

        self._buffer = buffer or FrameBuffer()
        self._capture: cv2.VideoCapture | None = None
        self._state = StreamState.STOPPED
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        # Stream metadata
        self._frame_width = 0
        self._frame_height = 0
        self._source_fps = 0.0
        self._total_frames = 0
        self._current_frame = 0

        # Image sequence handling
        self._image_paths: list[Path] = []
        self._image_index = 0

        # Callbacks
        self._on_frame_callback: Callable[[BufferedFrame], None] | None = None
        self._on_error_callback: Callable[[str], None] | None = None

    def open(self) -> bool:
        """Open the video source."""
        try:
            if self.source_type == FrameSource.VIDEO_FILE:
                return self._open_video_file()
            elif self.source_type == FrameSource.WEBCAM:
                return self._open_webcam()
            elif self.source_type == FrameSource.RTSP_STREAM:
                return self._open_rtsp()
            elif self.source_type == FrameSource.IMAGE_SEQUENCE:
                return self._open_image_sequence()
            return False
        except Exception as e:
            self._state = StreamState.ERROR
            if self._on_error_callback:
                self._on_error_callback(str(e))
            return False

    def _open_video_file(self) -> bool:
        """Open a video file."""
        if not self.source_path:
            raise ValueError("source_path required for video file")

        self._capture = cv2.VideoCapture(self.source_path)
        if not self._capture.isOpened():
            raise ValueError(f"Failed to open video: {self.source_path}")

        self._frame_width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._frame_height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._source_fps = self._capture.get(cv2.CAP_PROP_FPS)
        self._total_frames = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))

        return True

    def _open_webcam(self) -> bool:
        """Open webcam."""
        self._capture = cv2.VideoCapture(self.webcam_id)
        if not self._capture.isOpened():
            raise ValueError(f"Failed to open webcam: {self.webcam_id}")

        self._frame_width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._frame_height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._source_fps = self._capture.get(cv2.CAP_PROP_FPS) or 30.0
        self._total_frames = -1  # Unknown for webcam

        return True

    def _open_rtsp(self) -> bool:
        """Open RTSP stream."""
        if not self.source_path:
            raise ValueError("source_path required for RTSP stream")

        self._capture = cv2.VideoCapture(self.source_path)
        if not self._capture.isOpened():
            raise ValueError(f"Failed to open RTSP stream: {self.source_path}")

        self._frame_width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._frame_height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._source_fps = self._capture.get(cv2.CAP_PROP_FPS) or 30.0
        self._total_frames = -1  # Unknown for stream

        return True

    def _open_image_sequence(self) -> bool:
        """Open image sequence."""
        if not self.source_path:
            raise ValueError("source_path required for image sequence")

        path = Path(self.source_path)
        if path.is_dir():
            extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            self._image_paths = sorted([
                p for p in path.iterdir()
                if p.suffix.lower() in extensions
            ])
        else:
            # Assume glob pattern
            import glob
            self._image_paths = sorted([Path(p) for p in glob.glob(self.source_path)])

        if not self._image_paths:
            raise ValueError(f"No images found at: {self.source_path}")

        # Get dimensions from first image
        first_img = cv2.imread(str(self._image_paths[0]))
        self._frame_height, self._frame_width = first_img.shape[:2]
        self._source_fps = self.target_fps or 30.0
        self._total_frames = len(self._image_paths)

        return True

    def close(self) -> None:
        """Close the video source."""
        self.stop()
        if self._capture:
            self._capture.release()
            self._capture = None
        self._state = StreamState.STOPPED

    def start(self) -> None:
        """Start reading frames in background thread."""
        if self._state == StreamState.RUNNING:
            return

        if not self._capture and self.source_type != FrameSource.IMAGE_SEQUENCE:
            if not self.open():
                return

        self._stop_event.clear()
        self._state = StreamState.RUNNING

        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop reading frames."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._state = StreamState.STOPPED

    def pause(self) -> None:
        """Pause reading frames."""
        if self._state == StreamState.RUNNING:
            self._state = StreamState.PAUSED

    def resume(self) -> None:
        """Resume reading frames."""
        if self._state == StreamState.PAUSED:
            self._state = StreamState.RUNNING

    def _read_loop(self) -> None:
        """Main read loop running in background thread."""
        frame_interval = 1.0 / (self.target_fps or self._source_fps or 30.0)

        while not self._stop_event.is_set():
            if self._state == StreamState.PAUSED:
                time.sleep(0.01)
                continue

            start_time = time.time()

            try:
                frame = self._read_next_frame()
                if frame is None:
                    if self.source_type == FrameSource.VIDEO_FILE:
                        self._state = StreamState.FINISHED
                        break
                    elif self.source_type == FrameSource.IMAGE_SEQUENCE:
                        self._state = StreamState.FINISHED
                        break
                    continue

                # Add to buffer
                buffered = self._buffer.add_frame(frame, time.time())

                # Callback
                if self._on_frame_callback:
                    self._on_frame_callback(buffered)

                self._current_frame += 1

            except Exception as e:
                if self._on_error_callback:
                    self._on_error_callback(str(e))
                continue

            # Rate limiting
            elapsed = time.time() - start_time
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _read_next_frame(self) -> np.ndarray | None:
        """Read the next frame from source."""
        if self.source_type == FrameSource.IMAGE_SEQUENCE:
            return self._read_next_image()

        if not self._capture:
            return None

        ret, frame = self._capture.read()
        if not ret:
            return None

        # Convert BGR to RGB
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def _read_next_image(self) -> np.ndarray | None:
        """Read next image from sequence."""
        if self._image_index >= len(self._image_paths):
            return None

        path = self._image_paths[self._image_index]
        self._image_index += 1

        frame = cv2.imread(str(path))
        if frame is None:
            return None

        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def read_single_frame(self) -> BufferedFrame | None:
        """Read a single frame synchronously."""
        frame = self._read_next_frame()
        if frame is None:
            return None

        self._current_frame += 1
        return self._buffer.add_frame(frame, time.time())

    def iter_frames(self) -> Iterator[BufferedFrame]:
        """Iterate over all frames synchronously."""
        while True:
            frame = self.read_single_frame()
            if frame is None:
                break
            yield frame

    def seek(self, frame_number: int) -> bool:
        """Seek to a specific frame (video files only)."""
        if self.source_type == FrameSource.IMAGE_SEQUENCE:
            if 0 <= frame_number < len(self._image_paths):
                self._image_index = frame_number
                self._current_frame = frame_number
                return True
            return False

        if not self._capture:
            return False

        self._capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        self._current_frame = frame_number
        return True

    def set_on_frame_callback(self, callback: Callable[[BufferedFrame], None]) -> None:
        """Set callback for new frames."""
        self._on_frame_callback = callback

    def set_on_error_callback(self, callback: Callable[[str], None]) -> None:
        """Set callback for errors."""
        self._on_error_callback = callback

    @property
    def buffer(self) -> FrameBuffer:
        return self._buffer

    @property
    def state(self) -> StreamState:
        return self._state

    @property
    def is_running(self) -> bool:
        return self._state == StreamState.RUNNING

    @property
    def is_finished(self) -> bool:
        return self._state == StreamState.FINISHED

    @property
    def frame_width(self) -> int:
        return self._frame_width

    @property
    def frame_height(self) -> int:
        return self._frame_height

    @property
    def fps(self) -> float:
        return self._source_fps

    @property
    def total_frames(self) -> int:
        return self._total_frames

    @property
    def current_frame(self) -> int:
        return self._current_frame

    @property
    def progress(self) -> float:
        if self._total_frames <= 0:
            return 0.0
        return self._current_frame / self._total_frames

    def get_info(self) -> dict:
        """Get stream information."""
        return {
            "source_type": self.source_type.value,
            "source_path": self.source_path,
            "state": self._state.value,
            "dimensions": f"{self._frame_width}x{self._frame_height}",
            "fps": round(self._source_fps, 2),
            "total_frames": self._total_frames,
            "current_frame": self._current_frame,
            "progress": f"{self.progress:.1%}" if self._total_frames > 0 else "N/A",
        }

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
