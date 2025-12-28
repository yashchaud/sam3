"""Geometry extraction from segmentation masks."""

from dataclasses import dataclass
import numpy as np
import cv2

from ..models import GeometryProperties


@dataclass
class GeometryConfig:
    """Configuration for geometry extraction."""
    min_area_pixels: int = 10
    contour_approximation: int = cv2.CHAIN_APPROX_SIMPLE


class MaskGeometryExtractor:
    """Extracts geometric properties from segmentation masks."""

    def __init__(self, config: GeometryConfig | None = None):
        self.config = config or GeometryConfig()

    def extract(self, mask: np.ndarray) -> GeometryProperties:
        """
        Extract geometry properties from a binary mask.

        Args:
            mask: Binary mask (H, W)

        Returns:
            GeometryProperties
        """
        # Ensure binary
        binary = (mask > 0).astype(np.uint8)

        # Find contours
        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            self.config.contour_approximation,
        )

        if not contours:
            return self._empty_geometry()

        # Use largest contour
        contour = max(contours, key=cv2.contourArea)

        # Area and perimeter
        area = cv2.contourArea(contour)
        if area < self.config.min_area_pixels:
            return self._empty_geometry()

        perimeter = cv2.arcLength(contour, closed=True)

        # Centroid from moments
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            return self._empty_geometry()

        centroid_x = moments["m10"] / moments["m00"]
        centroid_y = moments["m01"] / moments["m00"]

        # Fit ellipse for orientation and dimensions
        length = 0.0
        width = 0.0
        orientation = 0.0

        if len(contour) >= 5:
            try:
                ellipse = cv2.fitEllipse(contour)
                (_, (minor_axis, major_axis), angle) = ellipse

                length = max(major_axis, minor_axis)
                width = min(major_axis, minor_axis)

                # Normalize angle to [-90, 90]
                orientation = angle
                if orientation > 90:
                    orientation -= 180
            except cv2.error:
                # Fallback to bounding rect
                _, _, w, h = cv2.boundingRect(contour)
                length = max(w, h)
                width = min(w, h)
        else:
            # Fallback to bounding rect
            _, _, w, h = cv2.boundingRect(contour)
            length = max(w, h)
            width = min(w, h)

        # Aspect ratio
        aspect_ratio = length / width if width > 0 else 1.0

        # Solidity (convex hull ratio)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0.0

        return GeometryProperties(
            area_pixels=int(area),
            perimeter_pixels=perimeter,
            length_pixels=length,
            width_pixels=width,
            orientation_degrees=orientation,
            aspect_ratio=aspect_ratio,
            solidity=solidity,
            centroid_x=centroid_x,
            centroid_y=centroid_y,
        )

    def extract_batch(self, masks: list[np.ndarray]) -> list[GeometryProperties]:
        """Extract geometry from multiple masks."""
        return [self.extract(mask) for mask in masks]

    def _empty_geometry(self) -> GeometryProperties:
        """Return empty geometry properties."""
        return GeometryProperties(
            area_pixels=0,
            perimeter_pixels=0.0,
            length_pixels=0.0,
            width_pixels=0.0,
            orientation_degrees=0.0,
            aspect_ratio=1.0,
            solidity=0.0,
            centroid_x=0.0,
            centroid_y=0.0,
        )
