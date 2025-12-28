"""
Core data models for the anomaly detection pipeline.

All data structures are immutable dataclasses to ensure
predictable data flow and easy debugging.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from pathlib import Path
import numpy as np
from datetime import datetime
import uuid


class DetectionType(Enum):
    """
    Classification of detected objects.

    STRUCTURE: Structural elements (beams, columns, walls, pipes, etc.)
    ANOMALY: Potential defects (cracks, corrosion, deformation, etc.)
    """
    STRUCTURE = "structure"
    ANOMALY = "anomaly"


@dataclass(frozen=True)
class BoundingBox:
    """
    Axis-aligned bounding box in pixel coordinates.

    Uses (x_min, y_min, x_max, y_max) format for consistency
    with most detection frameworks.
    """
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __post_init__(self):
        if self.x_min > self.x_max:
            raise ValueError(f"x_min ({self.x_min}) > x_max ({self.x_max})")
        if self.y_min > self.y_max:
            raise ValueError(f"y_min ({self.y_min}) > y_max ({self.y_max})")

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return (
            (self.x_min + self.x_max) / 2,
            (self.y_min + self.y_max) / 2
        )

    def to_xyxy(self) -> tuple[float, float, float, float]:
        """Return as (x_min, y_min, x_max, y_max) tuple."""
        return (self.x_min, self.y_min, self.x_max, self.y_max)

    def to_xywh(self) -> tuple[float, float, float, float]:
        """Return as (x_min, y_min, width, height) tuple."""
        return (self.x_min, self.y_min, self.width, self.height)

    def iou(self, other: "BoundingBox") -> float:
        """Compute Intersection over Union with another box."""
        inter_x_min = max(self.x_min, other.x_min)
        inter_y_min = max(self.y_min, other.y_min)
        inter_x_max = min(self.x_max, other.x_max)
        inter_y_max = min(self.y_max, other.y_max)

        if inter_x_min >= inter_x_max or inter_y_min >= inter_y_max:
            return 0.0

        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        union_area = self.area + other.area - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def contains_point(self, x: float, y: float) -> bool:
        """Check if point (x, y) is inside the box."""
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max


@dataclass(frozen=True)
class Detection:
    """
    A single detection from RT-DETR.

    Represents either a structural element or an anomaly candidate
    before segmentation refinement.
    """
    detection_id: str
    detection_type: DetectionType
    class_name: str  # e.g., "crack", "corrosion", "beam", "column"
    confidence: float
    bbox: BoundingBox

    def __post_init__(self):
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")


@dataclass
class SegmentationMask:
    """
    Binary segmentation mask from SAM3.

    The mask is stored as a numpy array where:
    - 1 (True) = anomaly pixel
    - 0 (False) = background pixel

    For memory efficiency in production, masks can optionally
    be saved to disk and loaded on demand.
    """
    mask: np.ndarray  # Shape: (H, W), dtype: bool or uint8
    mask_path: Optional[Path] = None  # If saved to disk
    sam_score: float = 0.0  # SAM's internal quality score

    def __post_init__(self):
        if self.mask.ndim != 2:
            raise ValueError(f"Mask must be 2D, got shape {self.mask.shape}")

    @property
    def height(self) -> int:
        return self.mask.shape[0]

    @property
    def width(self) -> int:
        return self.mask.shape[1]

    def pixel_count(self) -> int:
        """Number of positive pixels in mask."""
        return int(np.sum(self.mask > 0))

    def to_binary(self) -> np.ndarray:
        """Return mask as boolean array."""
        return self.mask.astype(bool)


@dataclass(frozen=True)
class GeometryProperties:
    """
    Basic geometric properties extracted from a segmentation mask.

    All measurements are in pixels. Conversion to real-world units
    requires camera calibration (out of scope for Phase-1).
    """
    area_pixels: int  # Total pixel count
    perimeter_pixels: float  # Contour length
    length_pixels: float  # Major axis of fitted ellipse
    width_pixels: float  # Minor axis of fitted ellipse
    orientation_degrees: float  # Angle of major axis, [-90, 90]
    aspect_ratio: float  # length / width
    solidity: float  # area / convex_hull_area, measures compactness
    centroid: tuple[float, float]  # (x, y) center of mass

    def __post_init__(self):
        if self.area_pixels < 0:
            raise ValueError(f"Area cannot be negative: {self.area_pixels}")
        if self.aspect_ratio < 0:
            raise ValueError(f"Aspect ratio cannot be negative: {self.aspect_ratio}")


@dataclass(frozen=True)
class AnomalyResult:
    """
    A fully processed anomaly with all associated data.

    This is the primary output unit of the Phase-1 pipeline.
    """
    # Identifiers
    anomaly_id: str
    frame_id: str
    timestamp: datetime

    # Classification
    defect_type: str  # e.g., "crack", "corrosion", "spalling"
    structure_type: Optional[str]  # e.g., "beam", "column", None if unassociated
    structure_id: Optional[str]  # ID of associated structure detection

    # Spatial data
    bbox: BoundingBox
    mask: SegmentationMask
    geometry: GeometryProperties

    # Confidence scores
    detection_confidence: float  # From RT-DETR
    segmentation_confidence: float  # From SAM3
    association_confidence: Optional[float]  # From structure matching

    @property
    def combined_confidence(self) -> float:
        """
        Aggregate confidence score.

        Simple geometric mean of available confidences.
        More sophisticated fusion can be added in later phases.
        """
        scores = [self.detection_confidence, self.segmentation_confidence]
        if self.association_confidence is not None:
            scores.append(self.association_confidence)

        product = 1.0
        for s in scores:
            product *= s
        return product ** (1.0 / len(scores))


@dataclass
class PipelineOutput:
    """
    Complete output from the Phase-1 pipeline for a single image/frame.
    """
    # Input metadata
    frame_id: str
    source_path: Optional[Path]
    image_width: int
    image_height: int
    timestamp: datetime

    # Results
    anomalies: list[AnomalyResult] = field(default_factory=list)
    structures: list[Detection] = field(default_factory=list)  # For reference

    # Processing metadata
    processing_time_ms: float = 0.0
    detector_time_ms: float = 0.0
    segmenter_time_ms: float = 0.0

    @property
    def anomaly_count(self) -> int:
        return len(self.anomalies)

    def to_dict(self) -> dict:
        """
        Serialize to dictionary for JSON export.

        Note: Masks are not included directly; use mask_path if available.
        """
        return {
            "frame_id": self.frame_id,
            "source_path": str(self.source_path) if self.source_path else None,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "timestamp": self.timestamp.isoformat(),
            "anomaly_count": self.anomaly_count,
            "anomalies": [
                {
                    "anomaly_id": a.anomaly_id,
                    "defect_type": a.defect_type,
                    "structure_type": a.structure_type,
                    "structure_id": a.structure_id,
                    "bbox": a.bbox.to_xyxy(),
                    "mask_path": str(a.mask.mask_path) if a.mask.mask_path else None,
                    "geometry": {
                        "area_pixels": a.geometry.area_pixels,
                        "perimeter_pixels": a.geometry.perimeter_pixels,
                        "length_pixels": a.geometry.length_pixels,
                        "width_pixels": a.geometry.width_pixels,
                        "orientation_degrees": a.geometry.orientation_degrees,
                        "aspect_ratio": a.geometry.aspect_ratio,
                        "solidity": a.geometry.solidity,
                        "centroid": a.geometry.centroid,
                    },
                    "detection_confidence": a.detection_confidence,
                    "segmentation_confidence": a.segmentation_confidence,
                    "association_confidence": a.association_confidence,
                    "combined_confidence": a.combined_confidence,
                }
                for a in self.anomalies
            ],
            "structures": [
                {
                    "detection_id": s.detection_id,
                    "class_name": s.class_name,
                    "confidence": s.confidence,
                    "bbox": s.bbox.to_xyxy(),
                }
                for s in self.structures
            ],
            "processing_time_ms": self.processing_time_ms,
        }


def generate_id(prefix: str = "") -> str:
    """Generate a unique ID with optional prefix."""
    short_uuid = str(uuid.uuid4())[:8]
    if prefix:
        return f"{prefix}_{short_uuid}"
    return short_uuid
