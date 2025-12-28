"""
Geometry extraction from segmentation masks.

Extracts basic geometric properties (area, perimeter, orientation, etc.)
from binary segmentation masks using OpenCV.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np

from anomaly_detection.models.data_models import (
    SegmentationMask,
    GeometryProperties,
)


@dataclass
class GeometryConfig:
    """
    Configuration for geometry extraction.

    Attributes:
        min_area_pixels: Minimum mask area to process
        contour_approximation: OpenCV contour approximation method
    """
    min_area_pixels: int = 10
    contour_approximation: int = 3  # cv2.CHAIN_APPROX_SIMPLE


class MaskGeometryExtractor:
    """
    Extracts geometric properties from segmentation masks.

    Uses OpenCV contour analysis and ellipse fitting to compute
    area, perimeter, orientation, and shape descriptors.

    Usage:
        extractor = MaskGeometryExtractor()
        properties = extractor.extract(mask)
        print(f"Area: {properties.area_pixels} pixels")
    """

    def __init__(self, config: Optional[GeometryConfig] = None):
        """
        Initialize the geometry extractor.

        Args:
            config: Geometry extraction configuration
        """
        self.config = config or GeometryConfig()

    def extract(self, mask: SegmentationMask) -> GeometryProperties:
        """
        Extract geometric properties from a segmentation mask.

        Args:
            mask: Binary segmentation mask

        Returns:
            GeometryProperties with all computed values
        """
        import cv2

        binary = mask.to_binary().astype(np.uint8) * 255

        # Find contours
        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            self.config.contour_approximation,
        )

        if not contours:
            return self._empty_geometry()

        # Use largest contour (main anomaly region)
        main_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(main_contour)

        if area < self.config.min_area_pixels:
            return self._empty_geometry()

        # Perimeter
        perimeter = cv2.arcLength(main_contour, closed=True)

        # Fit ellipse for orientation and dimensions
        # Requires at least 5 points
        if len(main_contour) >= 5:
            ellipse = cv2.fitEllipse(main_contour)
            center = ellipse[0]
            axes = ellipse[1]  # (width, height) of ellipse
            angle = ellipse[2]

            # Major/minor axes
            length = max(axes)
            width = min(axes)

            # Normalize angle to [-90, 90]
            orientation = self._normalize_angle(angle)
        else:
            # Fall back to bounding rect
            rect = cv2.minAreaRect(main_contour)
            center = rect[0]
            size = rect[1]
            angle = rect[2]

            length = max(size)
            width = min(size)
            orientation = self._normalize_angle(angle)

        # Aspect ratio (avoid division by zero)
        aspect_ratio = length / width if width > 0 else 1.0

        # Solidity = area / convex hull area
        hull = cv2.convexHull(main_contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0.0

        # Centroid from moments
        moments = cv2.moments(main_contour)
        if moments["m00"] > 0:
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]
        else:
            cx, cy = center

        return GeometryProperties(
            area_pixels=int(area),
            perimeter_pixels=float(perimeter),
            length_pixels=float(length),
            width_pixels=float(width),
            orientation_degrees=float(orientation),
            aspect_ratio=float(aspect_ratio),
            solidity=float(solidity),
            centroid=(float(cx), float(cy)),
        )

    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-90, 90] range."""
        while angle > 90:
            angle -= 180
        while angle < -90:
            angle += 180
        return angle

    def _empty_geometry(self) -> GeometryProperties:
        """Return geometry with zero values for empty masks."""
        return GeometryProperties(
            area_pixels=0,
            perimeter_pixels=0.0,
            length_pixels=0.0,
            width_pixels=0.0,
            orientation_degrees=0.0,
            aspect_ratio=1.0,
            solidity=0.0,
            centroid=(0.0, 0.0),
        )

    def extract_batch(
        self,
        masks: list[SegmentationMask],
    ) -> list[GeometryProperties]:
        """
        Extract geometry from multiple masks.

        Args:
            masks: List of segmentation masks

        Returns:
            List of GeometryProperties in same order
        """
        return [self.extract(mask) for mask in masks]
