"""Structure-defect association for linking anomalies to structures."""

from dataclasses import dataclass
from enum import Enum
import math

from ..models import Detection


class MatchStrategy(Enum):
    """Strategy for matching anomalies to structures."""
    CONTAINMENT = "containment"  # Anomaly center inside structure
    MAX_OVERLAP = "max_overlap"  # Highest IoU
    NEAREST = "nearest"          # Minimum centroid distance


@dataclass
class MatcherConfig:
    """Configuration for structure-defect matching."""
    strategy: MatchStrategy = MatchStrategy.CONTAINMENT
    min_overlap_threshold: float = 0.1
    max_distance_pixels: float = 100.0
    allow_unmatched: bool = True


@dataclass
class MatchResult:
    """Result of matching an anomaly to a structure."""
    anomaly_id: str
    structure_id: str | None
    structure_class: str | None
    confidence: float
    match_type: str  # "containment", "overlap", "proximity", "none"


class StructureDefectMatcher:
    """Matches anomalies to structural elements."""

    def __init__(self, config: MatcherConfig | None = None):
        self.config = config or MatcherConfig()

    def match(
        self,
        anomalies: list[Detection],
        structures: list[Detection],
    ) -> list[MatchResult]:
        """
        Match anomalies to structures.

        Args:
            anomalies: List of anomaly detections
            structures: List of structure detections

        Returns:
            List of MatchResult, one per anomaly
        """
        results = []

        for anomaly in anomalies:
            if self.config.strategy == MatchStrategy.CONTAINMENT:
                result = self._match_containment(anomaly, structures)
            elif self.config.strategy == MatchStrategy.MAX_OVERLAP:
                result = self._match_max_overlap(anomaly, structures)
            elif self.config.strategy == MatchStrategy.NEAREST:
                result = self._match_nearest(anomaly, structures)
            else:
                result = MatchResult(
                    anomaly_id=anomaly.detection_id,
                    structure_id=None,
                    structure_class=None,
                    confidence=0.0,
                    match_type="none",
                )

            results.append(result)

        return results

    def _match_containment(
        self,
        anomaly: Detection,
        structures: list[Detection],
    ) -> MatchResult:
        """Match by checking if anomaly center is inside structure."""
        cx, cy = anomaly.bbox.center

        for structure in structures:
            if structure.bbox.contains_point(cx, cy):
                return MatchResult(
                    anomaly_id=anomaly.detection_id,
                    structure_id=structure.detection_id,
                    structure_class=structure.class_name,
                    confidence=structure.confidence,
                    match_type="containment",
                )

        # Fallback to overlap if no containment
        return self._match_max_overlap(anomaly, structures)

    def _match_max_overlap(
        self,
        anomaly: Detection,
        structures: list[Detection],
    ) -> MatchResult:
        """Match by highest IoU."""
        best_structure = None
        best_iou = 0.0

        for structure in structures:
            iou = anomaly.bbox.iou(structure.bbox)
            if iou > best_iou and iou >= self.config.min_overlap_threshold:
                best_iou = iou
                best_structure = structure

        if best_structure:
            return MatchResult(
                anomaly_id=anomaly.detection_id,
                structure_id=best_structure.detection_id,
                structure_class=best_structure.class_name,
                confidence=best_iou * best_structure.confidence,
                match_type="overlap",
            )

        # Fallback to nearest if no overlap
        if self.config.allow_unmatched:
            return self._match_nearest(anomaly, structures)

        return MatchResult(
            anomaly_id=anomaly.detection_id,
            structure_id=None,
            structure_class=None,
            confidence=0.0,
            match_type="none",
        )

    def _match_nearest(
        self,
        anomaly: Detection,
        structures: list[Detection],
    ) -> MatchResult:
        """Match by minimum centroid distance."""
        if not structures:
            return MatchResult(
                anomaly_id=anomaly.detection_id,
                structure_id=None,
                structure_class=None,
                confidence=0.0,
                match_type="none",
            )

        anomaly_center = anomaly.bbox.center
        best_structure = None
        best_distance = float('inf')

        for structure in structures:
            struct_center = structure.bbox.center
            distance = math.sqrt(
                (anomaly_center[0] - struct_center[0]) ** 2 +
                (anomaly_center[1] - struct_center[1]) ** 2
            )
            if distance < best_distance:
                best_distance = distance
                best_structure = structure

        if best_structure and best_distance <= self.config.max_distance_pixels:
            # Confidence decreases with distance
            proximity_score = 1.0 - (best_distance / self.config.max_distance_pixels)
            confidence = proximity_score * best_structure.confidence

            return MatchResult(
                anomaly_id=anomaly.detection_id,
                structure_id=best_structure.detection_id,
                structure_class=best_structure.class_name,
                confidence=confidence,
                match_type="proximity",
            )

        return MatchResult(
            anomaly_id=anomaly.detection_id,
            structure_id=None,
            structure_class=None,
            confidence=0.0,
            match_type="none",
        )
