"""Data models for VLM Judge module."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal
import time
from collections import deque
import statistics


class VLMProvider(Enum):
    """Supported VLM providers."""
    LOCAL_QWEN = "local_qwen"
    OPENROUTER = "openrouter"


class PredictionType(Enum):
    """Type of prediction from VLM."""
    POINT = "point"
    BOX = "box"


@dataclass
class GridConfig:
    """Configuration for grid overlay on images."""
    cols: int = 3
    rows: int = 3
    line_color: tuple[int, int, int] = (255, 255, 255)
    line_thickness: int = 2
    label_font_scale: float = 0.6
    label_color: tuple[int, int, int] = (255, 255, 0)
    show_labels: bool = True

    @property
    def total_cells(self) -> int:
        return self.cols * self.rows


@dataclass
class VLMConfig:
    """Configuration for VLM Judge."""
    provider: VLMProvider = VLMProvider.LOCAL_QWEN

    # Local Qwen config
    qwen_model_path: str | None = None
    qwen_device: str = "cuda"

    # OpenRouter config
    openrouter_api_key: str | None = None
    openrouter_model: str = "qwen/qwen-2.5-vl-72b-instruct"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # Processing config
    grid_config: GridConfig = field(default_factory=GridConfig)
    process_every_n_frames: int = 10
    max_generation_frames: int = 60
    timeout_seconds: float = 2.0

    # Output preferences
    prefer_boxes: bool = True
    max_predictions_per_frame: int = 10
    min_confidence: float = 0.3


@dataclass(frozen=True)
class VLMPrediction:
    """Single prediction from VLM."""
    prediction_type: PredictionType
    confidence: float
    defect_type: str

    # For point predictions
    point: tuple[int, int] | None = None

    # For box predictions
    box: tuple[int, int, int, int] | None = None  # x_min, y_min, x_max, y_max

    # Grid cell reference (for debugging)
    grid_cell: tuple[int, int] | None = None  # col, row

    def get_center_point(self) -> tuple[int, int]:
        """Get center point for SAM prompting."""
        if self.point is not None:
            return self.point
        if self.box is not None:
            x_min, y_min, x_max, y_max = self.box
            return ((x_min + x_max) // 2, (y_min + y_max) // 2)
        raise ValueError("Prediction has neither point nor box")

    def get_sam_prompt(self) -> dict:
        """Get prompt format for SAM3."""
        if self.box is not None:
            return {
                "type": "box",
                "box": list(self.box),
                "label": self.defect_type,
            }
        if self.point is not None:
            return {
                "type": "point",
                "point": list(self.point),
                "label": 1,  # Positive point
            }
        raise ValueError("Prediction has neither point nor box")


@dataclass
class VLMResponse:
    """Response from VLM processing."""
    frame_id: str
    predictions: list[VLMPrediction]
    generation_time_ms: float
    is_valid: bool = True
    error_message: str | None = None
    provider: VLMProvider = VLMProvider.LOCAL_QWEN

    # Tracking for discard logic
    request_frame_index: int = 0
    current_frame_index: int = 0

    @property
    def is_stale(self) -> bool:
        """Check if response should be discarded (scene changed)."""
        return (self.current_frame_index - self.request_frame_index) > 60


class LatencyStats:
    """Track latency statistics for VLM processing."""

    def __init__(self, window_size: int = 100):
        self._window_size = window_size
        self._latencies: deque[float] = deque(maxlen=window_size)
        self._total_requests = 0
        self._failed_requests = 0
        self._discarded_requests = 0
        self._start_time = time.time()

    def record_latency(self, latency_ms: float, success: bool = True, discarded: bool = False) -> None:
        """Record a latency measurement."""
        self._latencies.append(latency_ms)
        self._total_requests += 1
        if not success:
            self._failed_requests += 1
        if discarded:
            self._discarded_requests += 1

    @property
    def mean_latency_ms(self) -> float:
        if not self._latencies:
            return 0.0
        return statistics.mean(self._latencies)

    @property
    def median_latency_ms(self) -> float:
        if not self._latencies:
            return 0.0
        return statistics.median(self._latencies)

    @property
    def p95_latency_ms(self) -> float:
        if len(self._latencies) < 20:
            return max(self._latencies) if self._latencies else 0.0
        sorted_latencies = sorted(self._latencies)
        idx = int(len(sorted_latencies) * 0.95)
        return sorted_latencies[idx]

    @property
    def p99_latency_ms(self) -> float:
        if len(self._latencies) < 100:
            return max(self._latencies) if self._latencies else 0.0
        sorted_latencies = sorted(self._latencies)
        idx = int(len(sorted_latencies) * 0.99)
        return sorted_latencies[idx]

    @property
    def min_latency_ms(self) -> float:
        return min(self._latencies) if self._latencies else 0.0

    @property
    def max_latency_ms(self) -> float:
        return max(self._latencies) if self._latencies else 0.0

    @property
    def success_rate(self) -> float:
        if self._total_requests == 0:
            return 1.0
        return (self._total_requests - self._failed_requests) / self._total_requests

    @property
    def discard_rate(self) -> float:
        if self._total_requests == 0:
            return 0.0
        return self._discarded_requests / self._total_requests

    @property
    def requests_per_second(self) -> float:
        elapsed = time.time() - self._start_time
        if elapsed == 0:
            return 0.0
        return self._total_requests / elapsed

    def to_dict(self) -> dict:
        """Export stats as dictionary."""
        return {
            "total_requests": self._total_requests,
            "failed_requests": self._failed_requests,
            "discarded_requests": self._discarded_requests,
            "success_rate": f"{self.success_rate:.1%}",
            "discard_rate": f"{self.discard_rate:.1%}",
            "mean_latency_ms": round(self.mean_latency_ms, 2),
            "median_latency_ms": round(self.median_latency_ms, 2),
            "p95_latency_ms": round(self.p95_latency_ms, 2),
            "p99_latency_ms": round(self.p99_latency_ms, 2),
            "min_latency_ms": round(self.min_latency_ms, 2),
            "max_latency_ms": round(self.max_latency_ms, 2),
            "requests_per_second": round(self.requests_per_second, 3),
        }

    def reset(self) -> None:
        """Reset all statistics."""
        self._latencies.clear()
        self._total_requests = 0
        self._failed_requests = 0
        self._discarded_requests = 0
        self._start_time = time.time()
