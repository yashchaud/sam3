"""Core data models for anomaly detection pipeline."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import numpy as np
import time
import uuid


class DetectionType(Enum):
    """Type of detection."""
    STRUCTURE = "structure"
    ANOMALY = "anomaly"


@dataclass(frozen=True)
class BoundingBox:
    """Immutable bounding box coordinates."""
    x_min: int
    y_min: int
    x_max: int
    y_max: int

    def __post_init__(self):
        if self.x_min > self.x_max:
            object.__setattr__(self, 'x_min', self.x_max)
            object.__setattr__(self, 'x_max', self.x_min)
        if self.y_min > self.y_max:
            object.__setattr__(self, 'y_min', self.y_max)
            object.__setattr__(self, 'y_max', self.y_min)

    @property
    def width(self) -> int:
        return self.x_max - self.x_min

    @property
    def height(self) -> int:
        return self.y_max - self.y_min

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> tuple[int, int]:
        return (
            (self.x_min + self.x_max) // 2,
            (self.y_min + self.y_max) // 2,
        )

    def iou(self, other: "BoundingBox") -> float:
        """Calculate Intersection over Union with another box."""
        x_min = max(self.x_min, other.x_min)
        y_min = max(self.y_min, other.y_min)
        x_max = min(self.x_max, other.x_max)
        y_max = min(self.y_max, other.y_max)

        if x_min >= x_max or y_min >= y_max:
            return 0.0

        intersection = (x_max - x_min) * (y_max - y_min)
        union = self.area + other.area - intersection

        return intersection / union if union > 0 else 0.0

    def contains_point(self, x: int, y: int) -> bool:
        """Check if point is inside box."""
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max

    def to_xyxy(self) -> tuple[int, int, int, int]:
        """Return as (x_min, y_min, x_max, y_max)."""
        return (self.x_min, self.y_min, self.x_max, self.y_max)

    def to_xywh(self) -> tuple[int, int, int, int]:
        """Return as (x_min, y_min, width, height)."""
        return (self.x_min, self.y_min, self.width, self.height)

    def scale(self, factor: float) -> "BoundingBox":
        """Scale box by factor around center."""
        cx, cy = self.center
        half_w = int(self.width * factor / 2)
        half_h = int(self.height * factor / 2)
        return BoundingBox(
            x_min=cx - half_w,
            y_min=cy - half_h,
            x_max=cx + half_w,
            y_max=cy + half_h,
        )


@dataclass
class Detection:
    """Single detection from object detector."""
    detection_id: str
    detection_type: DetectionType
    class_name: str
    confidence: float
    bbox: BoundingBox

    def __post_init__(self):
        if not 0.0 <= self.confidence <= 1.0:
            self.confidence = max(0.0, min(1.0, self.confidence))


@dataclass
class SegmentationMask:
    """Binary segmentation mask."""
    data: np.ndarray  # (H, W) binary mask
    mask_path: Path | None = None
    sam_score: float | None = None

    @property
    def pixel_count(self) -> int:
        return int(np.sum(self.data > 0))

    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape[:2]

    def to_binary(self, threshold: float = 0.5) -> np.ndarray:
        """Return binary mask."""
        return (self.data > threshold).astype(np.uint8)


@dataclass(frozen=True)
class GeometryProperties:
    """Geometric properties extracted from mask."""
    area_pixels: int
    perimeter_pixels: float
    length_pixels: float
    width_pixels: float
    orientation_degrees: float  # -90 to 90
    aspect_ratio: float
    solidity: float
    centroid_x: float
    centroid_y: float

    def to_dict(self) -> dict:
        return {
            "area_pixels": self.area_pixels,
            "perimeter_pixels": round(self.perimeter_pixels, 2),
            "length_pixels": round(self.length_pixels, 2),
            "width_pixels": round(self.width_pixels, 2),
            "orientation_degrees": round(self.orientation_degrees, 2),
            "aspect_ratio": round(self.aspect_ratio, 3),
            "solidity": round(self.solidity, 3),
            "centroid": [round(self.centroid_x, 2), round(self.centroid_y, 2)],
        }


@dataclass
class AnomalyResult:
    """Complete result for a detected anomaly."""
    anomaly_id: str
    frame_id: str
    timestamp: float
    defect_type: str
    structure_type: str | None
    bbox: BoundingBox
    mask: SegmentationMask | None
    geometry: GeometryProperties | None

    detection_confidence: float = 0.0
    segmentation_confidence: float = 0.0
    association_confidence: float = 0.0

    @property
    def combined_confidence(self) -> float:
        """Geometric mean of all confidences."""
        scores = [self.detection_confidence]
        if self.segmentation_confidence > 0:
            scores.append(self.segmentation_confidence)
        if self.association_confidence > 0:
            scores.append(self.association_confidence)

        if not scores:
            return 0.0

        product = 1.0
        for s in scores:
            product *= s
        return product ** (1.0 / len(scores))

    def to_dict(self) -> dict:
        return {
            "anomaly_id": self.anomaly_id,
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "defect_type": self.defect_type,
            "structure_type": self.structure_type,
            "bbox": self.bbox.to_xyxy(),
            "geometry": self.geometry.to_dict() if self.geometry else None,
            "confidence": {
                "detection": round(self.detection_confidence, 3),
                "segmentation": round(self.segmentation_confidence, 3),
                "association": round(self.association_confidence, 3),
                "combined": round(self.combined_confidence, 3),
            },
        }


@dataclass
class PipelineOutput:
    """Output from processing a single image/frame."""
    anomalies: list[AnomalyResult] = field(default_factory=list)
    structures: list[Detection] = field(default_factory=list)

    frame_id: str = ""
    source_path: str | None = None
    image_width: int = 0
    image_height: int = 0
    timestamp: float = field(default_factory=time.time)

    processing_time_ms: float = 0.0
    detector_time_ms: float = 0.0
    segmenter_time_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "frame_id": self.frame_id,
            "source_path": self.source_path,
            "dimensions": [self.image_width, self.image_height],
            "timestamp": self.timestamp,
            "anomalies": [a.to_dict() for a in self.anomalies],
            "structures": [
                {
                    "class_name": s.class_name,
                    "confidence": round(s.confidence, 3),
                    "bbox": s.bbox.to_xyxy(),
                }
                for s in self.structures
            ],
            "timing": {
                "total_ms": round(self.processing_time_ms, 2),
                "detector_ms": round(self.detector_time_ms, 2),
                "segmenter_ms": round(self.segmenter_time_ms, 2),
            },
        }
