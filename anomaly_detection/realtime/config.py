"""Configuration for real-time video processing."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from ..vlm.models import VLMConfig, VLMProvider, GridConfig
from ..segmenter.sam3_segmenter import SegmenterConfig


class FrameSource(Enum):
    """Source type for video frames."""
    VIDEO_FILE = "video_file"
    WEBCAM = "webcam"
    RTSP_STREAM = "rtsp_stream"
    IMAGE_SEQUENCE = "image_sequence"


@dataclass
class RealtimeConfig:
    """Configuration for real-time video processing pipeline (VLM + SAM3)."""

    # Source configuration
    source_type: FrameSource = FrameSource.VIDEO_FILE
    source_path: str | None = None
    webcam_id: int = 0

    # Model paths
    segmenter_model_path: Path | None = None

    # VLM configuration (primary detection method)
    vlm_config: VLMConfig = field(default_factory=VLMConfig)

    # Processing configuration
    target_fps: float = 30.0
    vlm_process_every_n_frames: int = 10
    vlm_max_generation_frames: int = 60
    confidence_threshold: float = 0.3

    # Buffer configuration
    frame_buffer_size: int = 120  # 4 seconds at 30fps
    max_pending_vlm_requests: int = 3

    # Output configuration
    save_masks: bool = False
    mask_output_dir: Path | None = None
    save_annotated_frames: bool = False
    annotated_output_dir: Path | None = None

    # Device configuration
    device: str = "auto"

    # Performance tuning
    skip_frames_on_lag: bool = True
    max_processing_backlog: int = 5

    def get_segmenter_config(self) -> SegmenterConfig:
        """Build segmenter configuration."""
        if self.segmenter_model_path is None:
            raise ValueError("segmenter_model_path is required")
        return SegmenterConfig(
            model_path=self.segmenter_model_path,
            device=self.device,
        )

    def get_vlm_config(self) -> VLMConfig:
        """Build VLM configuration with overrides."""
        config = self.vlm_config
        config.process_every_n_frames = self.vlm_process_every_n_frames
        config.max_generation_frames = self.vlm_max_generation_frames
        return config


@dataclass
class ProcessingStats:
    """Statistics for real-time processing."""
    total_frames_processed: int = 0
    total_frames_dropped: int = 0
    total_detections: int = 0
    total_vlm_predictions: int = 0
    total_vlm_discarded: int = 0

    avg_fps: float = 0.0
    avg_detection_time_ms: float = 0.0
    avg_segmentation_time_ms: float = 0.0
    avg_total_frame_time_ms: float = 0.0

    vlm_stats: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "frames": {
                "processed": self.total_frames_processed,
                "dropped": self.total_frames_dropped,
            },
            "detections": self.total_detections,
            "vlm": {
                "predictions": self.total_vlm_predictions,
                "discarded": self.total_vlm_discarded,
                **self.vlm_stats,
            },
            "performance": {
                "avg_fps": round(self.avg_fps, 2),
                "avg_detection_time_ms": round(self.avg_detection_time_ms, 2),
                "avg_segmentation_time_ms": round(self.avg_segmentation_time_ms, 2),
                "avg_total_frame_time_ms": round(self.avg_total_frame_time_ms, 2),
            },
        }
