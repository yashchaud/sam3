"""Image tiling utilities for processing large images."""

from anomaly_detection.tiling.tiler import (
    TileInfo,
    TileConfig,
    ImageTiler,
)
from anomaly_detection.tiling.coordinator import TiledDetectionCoordinator
from anomaly_detection.tiling.temporal_tiler import (
    TemporalTileConfig,
    TemporalTileManager,
    FrameType,
    FrameStrategy,
    AccumulatedDetection,
    CycleResult,
)

__all__ = [
    # Static tiling
    "TileInfo",
    "TileConfig",
    "ImageTiler",
    "TiledDetectionCoordinator",
    # Temporal/video tiling
    "TemporalTileConfig",
    "TemporalTileManager",
    "FrameType",
    "FrameStrategy",
    "AccumulatedDetection",
    "CycleResult",
]
