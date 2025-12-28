"""
Image tiling utilities for large image processing.

This module handles splitting large images into overlapping tiles,
tracking coordinate transformations, and stitching results back together.

The key challenge solved here:
    - RF-DETR processes tiles and returns tile-local bounding boxes
    - SAM must receive the TILE IMAGE with LOCAL coordinates (not global)
    - Final masks must be transformed back to global coordinates
    - Overlapping detections must be deduplicated
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class TileInfo:
    """
    Metadata for a single tile, tracking its position and scale.

    This class enables coordinate transformation between:
        - tile-local coordinates (what RF-DETR outputs)
        - global coordinates (final output space)

    Attributes:
        tile_id: Unique identifier for this tile
        row: Row index in tile grid
        col: Column index in tile grid

        # Global position (top-left corner in original image)
        global_x: X offset in original image
        global_y: Y offset in original image

        # Original crop size (before any scaling)
        crop_width: Width of crop from original image
        crop_height: Height of crop from original image

        # Tile size (after scaling, what RF-DETR sees)
        tile_width: Width of tile sent to detector
        tile_height: Height of tile sent to detector

        # Scale factors (tile_size / crop_size)
        scale_x: Horizontal scale factor
        scale_y: Vertical scale factor

        # The actual tile image data
        tile_image: The tile numpy array (H, W, C)
    """
    tile_id: str
    row: int
    col: int

    # Position in global image
    global_x: int
    global_y: int

    # Original crop dimensions
    crop_width: int
    crop_height: int

    # Tile dimensions (after scaling)
    tile_width: int
    tile_height: int

    # Scale factors
    scale_x: float = 1.0
    scale_y: float = 1.0

    # Tile image data
    tile_image: Optional[np.ndarray] = field(default=None, repr=False)

    def local_to_global_bbox(
        self,
        x_min: float,
        y_min: float,
        x_max: float,
        y_max: float,
    ) -> tuple[float, float, float, float]:
        """
        Transform tile-local bounding box to global coordinates.

        This handles:
            1. Inverse scaling (if tile was resized)
            2. Translation (add tile offset)

        Args:
            x_min, y_min, x_max, y_max: Tile-local coordinates

        Returns:
            (global_x_min, global_y_min, global_x_max, global_y_max)
        """
        # Step 1: Inverse scale to get crop-space coordinates
        crop_x_min = x_min / self.scale_x
        crop_y_min = y_min / self.scale_y
        crop_x_max = x_max / self.scale_x
        crop_y_max = y_max / self.scale_y

        # Step 2: Translate to global coordinates
        global_x_min = crop_x_min + self.global_x
        global_y_min = crop_y_min + self.global_y
        global_x_max = crop_x_max + self.global_x
        global_y_max = crop_y_max + self.global_y

        return global_x_min, global_y_min, global_x_max, global_y_max

    def global_to_local_bbox(
        self,
        x_min: float,
        y_min: float,
        x_max: float,
        y_max: float,
    ) -> tuple[float, float, float, float]:
        """
        Transform global bounding box to tile-local coordinates.

        Inverse of local_to_global_bbox.

        Args:
            x_min, y_min, x_max, y_max: Global coordinates

        Returns:
            (local_x_min, local_y_min, local_x_max, local_y_max)
        """
        # Step 1: Translate to crop-space (relative to tile origin)
        crop_x_min = x_min - self.global_x
        crop_y_min = y_min - self.global_y
        crop_x_max = x_max - self.global_x
        crop_y_max = y_max - self.global_y

        # Step 2: Apply scale to get tile-space coordinates
        local_x_min = crop_x_min * self.scale_x
        local_y_min = crop_y_min * self.scale_y
        local_x_max = crop_x_max * self.scale_x
        local_y_max = crop_y_max * self.scale_y

        return local_x_min, local_y_min, local_x_max, local_y_max

    def local_to_global_mask(self, local_mask: np.ndarray, global_shape: tuple[int, int]) -> np.ndarray:
        """
        Transform tile-local mask to global coordinates.

        Args:
            local_mask: Binary mask in tile coordinates (tile_height, tile_width)
            global_shape: (height, width) of the global image

        Returns:
            Binary mask in global coordinates (global_height, global_width)
        """
        import cv2

        # Create empty global mask
        global_mask = np.zeros(global_shape, dtype=np.uint8)

        # If mask needs unscaling, resize it first
        if self.scale_x != 1.0 or self.scale_y != 1.0:
            # Resize to crop dimensions (inverse of scaling)
            unscaled_mask = cv2.resize(
                local_mask,
                (self.crop_width, self.crop_height),
                interpolation=cv2.INTER_NEAREST,
            )
        else:
            unscaled_mask = local_mask

        # Calculate placement region in global mask
        y_start = self.global_y
        y_end = min(self.global_y + self.crop_height, global_shape[0])
        x_start = self.global_x
        x_end = min(self.global_x + self.crop_width, global_shape[1])

        # Calculate corresponding region in unscaled mask
        mask_y_end = y_end - y_start
        mask_x_end = x_end - x_start

        # Place mask in global coordinates
        global_mask[y_start:y_end, x_start:x_end] = unscaled_mask[:mask_y_end, :mask_x_end]

        return global_mask

    def local_point_to_global(self, x: float, y: float) -> tuple[float, float]:
        """Transform a single point from tile-local to global coordinates."""
        global_x = (x / self.scale_x) + self.global_x
        global_y = (y / self.scale_y) + self.global_y
        return global_x, global_y

    def global_point_to_local(self, x: float, y: float) -> tuple[float, float]:
        """Transform a single point from global to tile-local coordinates."""
        local_x = (x - self.global_x) * self.scale_x
        local_y = (y - self.global_y) * self.scale_y
        return local_x, local_y


@dataclass
class TileConfig:
    """
    Configuration for image tiling.

    Attributes:
        tile_size: Target tile size (width, height) for detector input
        overlap: Overlap between adjacent tiles in pixels (before scaling)
        min_tile_size: Minimum tile dimension (smaller images won't be tiled)
        scale_tiles: Whether to scale tiles to fixed size
        pad_tiles: Whether to pad edge tiles to full size
        pad_value: Padding value (0-255 for uint8)
    """
    tile_size: tuple[int, int] = (640, 640)  # RF-DETR optimal input
    overlap: int = 64  # Overlap to catch objects on tile boundaries
    min_tile_size: int = 320  # Don't create tiny edge tiles
    scale_tiles: bool = True  # Scale crops to tile_size
    pad_tiles: bool = True  # Pad edge tiles
    pad_value: int = 0  # Black padding

    def __post_init__(self):
        if self.overlap < 0:
            raise ValueError(f"overlap must be >= 0, got {self.overlap}")
        if self.tile_size[0] < 1 or self.tile_size[1] < 1:
            raise ValueError(f"tile_size must be positive, got {self.tile_size}")


class ImageTiler:
    """
    Splits large images into overlapping tiles for processing.

    Handles:
        - Grid-based tiling with configurable overlap
        - Optional scaling to fixed tile size
        - Edge padding for uniform tile dimensions
        - Coordinate transformation tracking

    Usage:
        config = TileConfig(tile_size=(640, 640), overlap=64)
        tiler = ImageTiler(config)

        # Generate tiles
        tiles = tiler.create_tiles(large_image)

        for tile_info in tiles:
            # Process tile with detector (uses tile_info.tile_image)
            detections = detector.detect(tile_info.tile_image)

            # Coordinates are tile-local, transform to global
            for det in detections:
                global_bbox = tile_info.local_to_global_bbox(
                    det.bbox.x_min, det.bbox.y_min,
                    det.bbox.x_max, det.bbox.y_max
                )

    Note:
        SAM should receive tile_info.tile_image with tile-local bounding boxes.
        Only transform to global coordinates AFTER segmentation.
    """

    def __init__(self, config: TileConfig):
        """
        Initialize the tiler.

        Args:
            config: Tiling configuration
        """
        self.config = config

    def should_tile(self, image: np.ndarray) -> bool:
        """
        Check if image is large enough to require tiling.

        Args:
            image: Input image (H, W, C)

        Returns:
            True if image should be tiled
        """
        h, w = image.shape[:2]
        target_w, target_h = self.config.tile_size

        # Tile if image is larger than tile size in either dimension
        return w > target_w or h > target_h

    def create_tiles(self, image: np.ndarray) -> list[TileInfo]:
        """
        Split image into overlapping tiles.

        Args:
            image: Input image as numpy array (H, W, C)

        Returns:
            List of TileInfo objects with tile images and metadata
        """
        h, w = image.shape[:2]
        target_w, target_h = self.config.tile_size
        overlap = self.config.overlap

        # If image fits in one tile, return single tile
        if not self.should_tile(image):
            return [self._create_single_tile(image)]

        # Calculate step size (tile size minus overlap)
        step_x = target_w - overlap
        step_y = target_h - overlap

        # Generate tile grid
        tiles = []
        tile_idx = 0

        row = 0
        y = 0
        while y < h:
            col = 0
            x = 0
            while x < w:
                # Calculate crop region
                crop_x_end = min(x + target_w, w)
                crop_y_end = min(y + target_h, h)
                crop_w = crop_x_end - x
                crop_h = crop_y_end - y

                # Skip tiny edge tiles
                if crop_w < self.config.min_tile_size or crop_h < self.config.min_tile_size:
                    x += step_x
                    col += 1
                    continue

                # Extract crop
                crop = image[y:crop_y_end, x:crop_x_end].copy()

                # Create tile (with optional scaling/padding)
                tile_info = self._create_tile(
                    crop=crop,
                    tile_id=f"tile_{tile_idx:03d}",
                    row=row,
                    col=col,
                    global_x=x,
                    global_y=y,
                )
                tiles.append(tile_info)

                tile_idx += 1
                x += step_x
                col += 1

            y += step_y
            row += 1

        return tiles

    def _create_single_tile(self, image: np.ndarray) -> TileInfo:
        """Create a single tile for small images."""
        h, w = image.shape[:2]
        target_w, target_h = self.config.tile_size

        # Optionally pad to target size
        if self.config.pad_tiles and (w < target_w or h < target_h):
            tile_image = self._pad_image(image, target_w, target_h)
            tile_w, tile_h = target_w, target_h
        else:
            tile_image = image.copy()
            tile_w, tile_h = w, h

        return TileInfo(
            tile_id="tile_000",
            row=0,
            col=0,
            global_x=0,
            global_y=0,
            crop_width=w,
            crop_height=h,
            tile_width=tile_w,
            tile_height=tile_h,
            scale_x=tile_w / w if w > 0 else 1.0,
            scale_y=tile_h / h if h > 0 else 1.0,
            tile_image=tile_image,
        )

    def _create_tile(
        self,
        crop: np.ndarray,
        tile_id: str,
        row: int,
        col: int,
        global_x: int,
        global_y: int,
    ) -> TileInfo:
        """Create a tile with optional scaling and padding."""
        crop_h, crop_w = crop.shape[:2]
        target_w, target_h = self.config.tile_size

        if self.config.scale_tiles:
            # Scale crop to target size
            tile_image = self._resize_image(crop, target_w, target_h)
            tile_w, tile_h = target_w, target_h
            scale_x = target_w / crop_w
            scale_y = target_h / crop_h
        elif self.config.pad_tiles:
            # Pad crop to target size (no scaling)
            tile_image = self._pad_image(crop, target_w, target_h)
            tile_w, tile_h = target_w, target_h
            scale_x = 1.0
            scale_y = 1.0
        else:
            # Use crop as-is
            tile_image = crop
            tile_w, tile_h = crop_w, crop_h
            scale_x = 1.0
            scale_y = 1.0

        return TileInfo(
            tile_id=tile_id,
            row=row,
            col=col,
            global_x=global_x,
            global_y=global_y,
            crop_width=crop_w,
            crop_height=crop_h,
            tile_width=tile_w,
            tile_height=tile_h,
            scale_x=scale_x,
            scale_y=scale_y,
            tile_image=tile_image,
        )

    def _resize_image(self, image: np.ndarray, width: int, height: int) -> np.ndarray:
        """Resize image to target dimensions."""
        import cv2
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)

    def _pad_image(self, image: np.ndarray, width: int, height: int) -> np.ndarray:
        """Pad image to target dimensions."""
        h, w = image.shape[:2]

        if w >= width and h >= height:
            return image

        # Create padded output
        if image.ndim == 3:
            padded = np.full((height, width, image.shape[2]), self.config.pad_value, dtype=image.dtype)
        else:
            padded = np.full((height, width), self.config.pad_value, dtype=image.dtype)

        # Copy original image to top-left
        padded[:h, :w] = image

        return padded

    def stitch_masks(
        self,
        tile_masks: list[tuple[TileInfo, np.ndarray]],
        global_shape: tuple[int, int],
    ) -> np.ndarray:
        """
        Combine tile masks into a single global mask.

        For overlapping regions, uses maximum value (union).

        Args:
            tile_masks: List of (TileInfo, local_mask) tuples
            global_shape: (height, width) of original image

        Returns:
            Combined global mask
        """
        global_mask = np.zeros(global_shape, dtype=np.uint8)

        for tile_info, local_mask in tile_masks:
            # Transform local mask to global coordinates
            tile_global_mask = tile_info.local_to_global_mask(local_mask, global_shape)

            # Combine with union (max)
            global_mask = np.maximum(global_mask, tile_global_mask)

        return global_mask
