"""
Structure-Defect association logic.

Associates detected anomalies with their parent structural elements
based on spatial relationships (containment, overlap, proximity).
"""

from dataclasses import dataclass
from typing import Optional
from enum import Enum

from anomaly_detection.models.data_models import Detection, BoundingBox


class MatchStrategy(Enum):
    """Strategy for matching anomalies to structures."""
    CONTAINMENT = "containment"  # Anomaly center must be inside structure
    MAX_OVERLAP = "max_overlap"  # Match to structure with highest IoU
    NEAREST = "nearest"  # Match to nearest structure by centroid distance


@dataclass
class MatcherConfig:
    """
    Configuration for structure-defect matching.

    Attributes:
        strategy: Matching strategy to use
        min_overlap_threshold: Minimum IoU for overlap-based matching
        max_distance_pixels: Maximum distance for proximity matching
        allow_unmatched: If False, raises error for unmatched anomalies
    """
    strategy: MatchStrategy = MatchStrategy.MAX_OVERLAP
    min_overlap_threshold: float = 0.1
    max_distance_pixels: float = 100.0
    allow_unmatched: bool = True


@dataclass
class MatchResult:
    """Result of matching a single anomaly to a structure."""
    anomaly_id: str
    structure_id: Optional[str]
    structure_class: Optional[str]
    confidence: float  # 0.0 if no match, otherwise based on IoU/distance
    match_type: str  # "containment", "overlap", "proximity", "none"


class StructureDefectMatcher:
    """
    Matches anomalies to their parent structural elements.

    The matcher uses spatial relationships between detection bounding boxes
    to associate each anomaly with the most likely containing/overlapping
    structural element.

    Usage:
        config = MatcherConfig(strategy=MatchStrategy.MAX_OVERLAP)
        matcher = StructureDefectMatcher(config)
        results = matcher.match(anomalies, structures)
    """

    def __init__(self, config: Optional[MatcherConfig] = None):
        """
        Initialize the matcher.

        Args:
            config: Matcher configuration. Uses defaults if None.
        """
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
            result = self._match_single(anomaly, structures)
            results.append(result)

        return results

    def _match_single(
        self,
        anomaly: Detection,
        structures: list[Detection],
    ) -> MatchResult:
        """
        Match a single anomaly to the best structure.

        Args:
            anomaly: Single anomaly detection
            structures: All available structure detections

        Returns:
            MatchResult for this anomaly
        """
        if not structures:
            return MatchResult(
                anomaly_id=anomaly.detection_id,
                structure_id=None,
                structure_class=None,
                confidence=0.0,
                match_type="none",
            )

        if self.config.strategy == MatchStrategy.CONTAINMENT:
            return self._match_by_containment(anomaly, structures)
        elif self.config.strategy == MatchStrategy.MAX_OVERLAP:
            return self._match_by_overlap(anomaly, structures)
        elif self.config.strategy == MatchStrategy.NEAREST:
            return self._match_by_proximity(anomaly, structures)
        else:
            raise ValueError(f"Unknown strategy: {self.config.strategy}")

    def _match_by_containment(
        self,
        anomaly: Detection,
        structures: list[Detection],
    ) -> MatchResult:
        """Match based on anomaly center being inside structure."""
        anomaly_center = anomaly.bbox.center

        containing = []
        for struct in structures:
            if struct.bbox.contains_point(*anomaly_center):
                containing.append(struct)

        if not containing:
            # Fall back to overlap if no containment
            return self._match_by_overlap(anomaly, structures)

        # If multiple contain the center, pick smallest (most specific)
        best = min(containing, key=lambda s: s.bbox.area)

        # Confidence based on how centered the anomaly is
        confidence = self._compute_containment_confidence(anomaly.bbox, best.bbox)

        return MatchResult(
            anomaly_id=anomaly.detection_id,
            structure_id=best.detection_id,
            structure_class=best.class_name,
            confidence=confidence,
            match_type="containment",
        )

    def _match_by_overlap(
        self,
        anomaly: Detection,
        structures: list[Detection],
    ) -> MatchResult:
        """Match based on maximum IoU overlap."""
        best_struct = None
        best_iou = 0.0

        for struct in structures:
            iou = anomaly.bbox.iou(struct.bbox)
            if iou > best_iou:
                best_iou = iou
                best_struct = struct

        if best_struct is None or best_iou < self.config.min_overlap_threshold:
            # Fall back to proximity if no sufficient overlap
            return self._match_by_proximity(anomaly, structures)

        return MatchResult(
            anomaly_id=anomaly.detection_id,
            structure_id=best_struct.detection_id,
            structure_class=best_struct.class_name,
            confidence=best_iou,  # IoU directly as confidence
            match_type="overlap",
        )

    def _match_by_proximity(
        self,
        anomaly: Detection,
        structures: list[Detection],
    ) -> MatchResult:
        """Match based on minimum centroid distance."""
        anomaly_center = anomaly.bbox.center

        best_struct = None
        best_distance = float("inf")

        for struct in structures:
            struct_center = struct.bbox.center
            distance = (
                (anomaly_center[0] - struct_center[0]) ** 2 +
                (anomaly_center[1] - struct_center[1]) ** 2
            ) ** 0.5

            if distance < best_distance:
                best_distance = distance
                best_struct = struct

        if best_struct is None or best_distance > self.config.max_distance_pixels:
            return MatchResult(
                anomaly_id=anomaly.detection_id,
                structure_id=None,
                structure_class=None,
                confidence=0.0,
                match_type="none",
            )

        # Convert distance to confidence (closer = higher confidence)
        confidence = max(0.0, 1.0 - (best_distance / self.config.max_distance_pixels))

        return MatchResult(
            anomaly_id=anomaly.detection_id,
            structure_id=best_struct.detection_id,
            structure_class=best_struct.class_name,
            confidence=confidence,
            match_type="proximity",
        )

    def _compute_containment_confidence(
        self,
        inner: BoundingBox,
        outer: BoundingBox,
    ) -> float:
        """
        Compute confidence for containment match.

        Higher confidence when anomaly is well-centered and smaller
        relative to structure.

        Args:
            inner: Anomaly bounding box
            outer: Structure bounding box

        Returns:
            Confidence score in [0, 1]
        """
        # IoU component
        iou = inner.iou(outer)

        # Size ratio component (anomaly should be smaller than structure)
        size_ratio = min(1.0, inner.area / outer.area) if outer.area > 0 else 0.0
        size_score = 1.0 - size_ratio  # Smaller anomaly = higher score

        # Combine (weighted average favoring IoU)
        return 0.7 * iou + 0.3 * size_score
