"""
Object detection module using RF-DETR.

RF-DETR is a state-of-the-art real-time detection transformer developed
by Roboflow. It achieves 60+ mAP on COCO with real-time performance.

License: Apache 2.0 (full commercial use permitted)
Source: https://github.com/roboflow/rf-detr

Available variants:
    - NANO:   48.4 mAP, 2.32ms (fastest)
    - SMALL:  53.0 mAP, 3.52ms
    - MEDIUM: 54.7 mAP, 4.52ms (recommended)
    - LARGE:  56.7 mAP, 9.64ms (best accuracy)
"""

from anomaly_detection.detector.rf_detr_detector import (
    RFDETRDetector,
    DetectorConfig,
    DetectorOutput,
    RFDETRVariant,
    DEFAULT_STRUCTURE_CLASSES,
    DEFAULT_ANOMALY_CLASSES,
)

__all__ = [
    "RFDETRDetector",
    "DetectorConfig",
    "DetectorOutput",
    "RFDETRVariant",
    "DEFAULT_STRUCTURE_CLASSES",
    "DEFAULT_ANOMALY_CLASSES",
]
