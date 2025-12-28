"""Anomaly Detection Pipeline for structural inspection."""

from .models import (
    DetectionType,
    BoundingBox,
    Detection,
    SegmentationMask,
    GeometryProperties,
    AnomalyResult,
    PipelineOutput,
)
from .config import EnvironmentConfig

__version__ = "0.2.0"

__all__ = [
    "DetectionType",
    "BoundingBox",
    "Detection",
    "SegmentationMask",
    "GeometryProperties",
    "AnomalyResult",
    "PipelineOutput",
    "EnvironmentConfig",
]
