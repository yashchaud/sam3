"""Object detection module."""

from .rf_detr_detector import (
    RFDETRVariant,
    DetectorConfig,
    DetectorOutput,
    RFDETRDetector,
)

__all__ = [
    "RFDETRVariant",
    "DetectorConfig",
    "DetectorOutput",
    "RFDETRDetector",
]
