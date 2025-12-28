"""Frame buffer for real-time video processing."""

from dataclasses import dataclass, field
from collections import deque
import time
import threading
import numpy as np
from typing import Iterator


@dataclass
class BufferedFrame:
    """A single buffered frame with metadata."""
    image: np.ndarray
    frame_index: int
    timestamp: float  # Capture timestamp
    frame_id: str

    # Processing state
    is_processed: bool = False
    processing_start_time: float | None = None
    processing_end_time: float | None = None

    @property
    def processing_time_ms(self) -> float | None:
        if self.processing_start_time and self.processing_end_time:
            return (self.processing_end_time - self.processing_start_time) * 1000
        return None

    @property
    def age_ms(self) -> float:
        """Time since frame was captured."""
        return (time.time() - self.timestamp) * 1000


class FrameBuffer:
    """
    Thread-safe circular buffer for video frames.

    Supports:
    - Adding frames from capture thread
    - Getting frames for processing
    - Dropping old frames when processing is too slow
    """

    def __init__(self, max_size: int = 120):
        self._buffer: deque[BufferedFrame] = deque(maxlen=max_size)
        self._lock = threading.Lock()
        self._frame_counter = 0
        self._dropped_count = 0
        self._start_time = time.time()

    def add_frame(self, image: np.ndarray, timestamp: float | None = None) -> BufferedFrame:
        """
        Add a new frame to the buffer.

        Args:
            image: Frame image (RGB, HWC)
            timestamp: Capture timestamp (default: current time)

        Returns:
            The buffered frame object
        """
        with self._lock:
            frame = BufferedFrame(
                image=image,
                frame_index=self._frame_counter,
                timestamp=timestamp or time.time(),
                frame_id=f"frame_{self._frame_counter:08d}",
            )
            self._frame_counter += 1

            # Track if we're dropping frames
            if len(self._buffer) == self._buffer.maxlen:
                oldest = self._buffer[0]
                if not oldest.is_processed:
                    self._dropped_count += 1

            self._buffer.append(frame)
            return frame

    def get_latest_unprocessed(self) -> BufferedFrame | None:
        """Get the most recent unprocessed frame."""
        with self._lock:
            for frame in reversed(self._buffer):
                if not frame.is_processed:
                    return frame
            return None

    def get_next_unprocessed(self) -> BufferedFrame | None:
        """Get the oldest unprocessed frame."""
        with self._lock:
            for frame in self._buffer:
                if not frame.is_processed:
                    return frame
            return None

    def get_frame_by_index(self, frame_index: int) -> BufferedFrame | None:
        """Get a specific frame by index."""
        with self._lock:
            for frame in self._buffer:
                if frame.frame_index == frame_index:
                    return frame
            return None

    def mark_processed(self, frame_index: int) -> None:
        """Mark a frame as processed."""
        with self._lock:
            for frame in self._buffer:
                if frame.frame_index == frame_index:
                    frame.is_processed = True
                    frame.processing_end_time = time.time()
                    break

    def mark_processing_started(self, frame_index: int) -> None:
        """Mark that processing has started for a frame."""
        with self._lock:
            for frame in self._buffer:
                if frame.frame_index == frame_index:
                    frame.processing_start_time = time.time()
                    break

    def get_unprocessed_frames(self, max_count: int = 10) -> list[BufferedFrame]:
        """Get multiple unprocessed frames."""
        with self._lock:
            frames = []
            for frame in self._buffer:
                if not frame.is_processed:
                    frames.append(frame)
                    if len(frames) >= max_count:
                        break
            return frames

    def get_frames_for_vlm(
        self,
        every_n: int = 10,
        max_pending: int = 3,
    ) -> list[BufferedFrame]:
        """
        Get frames that should be sent to VLM.

        Args:
            every_n: Process every N frames
            max_pending: Maximum pending requests

        Returns:
            List of frames to process
        """
        with self._lock:
            candidates = []
            for frame in self._buffer:
                if frame.frame_index % every_n == 0:
                    candidates.append(frame)

            return candidates[-max_pending:]

    def drop_old_unprocessed(self, max_age_ms: float = 2000) -> int:
        """
        Drop frames older than max_age_ms that haven't been processed.

        Returns:
            Number of frames dropped
        """
        with self._lock:
            dropped = 0
            current_time = time.time()

            for frame in list(self._buffer):
                if not frame.is_processed:
                    age_ms = (current_time - frame.timestamp) * 1000
                    if age_ms > max_age_ms:
                        frame.is_processed = True  # Mark as "processed" to skip
                        dropped += 1
                        self._dropped_count += 1

            return dropped

    def clear(self) -> None:
        """Clear all frames from buffer."""
        with self._lock:
            self._buffer.clear()

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._buffer)

    @property
    def unprocessed_count(self) -> int:
        with self._lock:
            return sum(1 for f in self._buffer if not f.is_processed)

    @property
    def frame_count(self) -> int:
        return self._frame_counter

    @property
    def dropped_count(self) -> int:
        return self._dropped_count

    @property
    def current_fps(self) -> float:
        elapsed = time.time() - self._start_time
        if elapsed == 0:
            return 0.0
        return self._frame_counter / elapsed

    def get_stats(self) -> dict:
        """Get buffer statistics."""
        with self._lock:
            return {
                "size": len(self._buffer),
                "max_size": self._buffer.maxlen,
                "total_frames": self._frame_counter,
                "dropped_frames": self._dropped_count,
                "unprocessed_count": sum(1 for f in self._buffer if not f.is_processed),
                "current_fps": round(self.current_fps, 2),
            }

    def __len__(self) -> int:
        return self.size

    def __iter__(self) -> Iterator[BufferedFrame]:
        with self._lock:
            return iter(list(self._buffer))
