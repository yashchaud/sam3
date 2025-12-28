"""Pipeline orchestration module."""

from anomaly_detection.pipeline.phase1_pipeline import (
    Phase1Pipeline,
    PipelineConfig,
)
from anomaly_detection.pipeline.video_processor import (
    VideoProcessor,
    VideoConfig,
    CycleOutput,
)
from anomaly_detection.detector.rf_detr_detector import RFDETRVariant
from anomaly_detection.tiling.tiler import TileConfig
from anomaly_detection.tiling.temporal_tiler import TemporalTileConfig

__all__ = [
    # Image pipeline
    "Phase1Pipeline",
    "PipelineConfig",
    # Video pipeline
    "VideoProcessor",
    "VideoConfig",
    "CycleOutput",
    # Configuration
    "RFDETRVariant",
    "TileConfig",
    "TemporalTileConfig",
]
