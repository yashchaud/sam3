"""Image tiling utilities for large image processing."""

from dataclasses import dataclass
import numpy as np

from ..models import BoundingBox


@dataclass
class TileConfig:
    """Configuration for image tiling."""
    tile_size: tuple[int, int] = (640, 640)
    overlap: int = 64
    min_tile_size: int = 320
    scale_tiles: bool = False
    pad_tiles: bool = True


@dataclass
class TileInfo:
    """Information about a single tile."""
    tile_id: int
    image: np.ndarray
    x_offset: int
    y_offset: int
    original_width: int
    original_height: int
    scale_factor: float = 1.0

    @property
    def width(self) -> int:
        return self.image.shape[1]

    @property
    def height(self) -> int:
        return self.image.shape[0]

    def local_to_global_bbox(self, bbox: BoundingBox) -> BoundingBox:
        """Transform tile-local bbox to global coordinates."""
        # Account for scaling if tile was resized
        if self.scale_factor != 1.0:
            inv_scale = 1.0 / self.scale_factor
            x_min = int(bbox.x_min * inv_scale)
            y_min = int(bbox.y_min * inv_scale)
            x_max = int(bbox.x_max * inv_scale)
            y_max = int(bbox.y_max * inv_scale)
        else:
            x_min, y_min, x_max, y_max = bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max

        # Add offset
        return BoundingBox(
            x_min=x_min + self.x_offset,
            y_min=y_min + self.y_offset,
            x_max=x_max + self.x_offset,
            y_max=y_max + self.y_offset,
        )

    def global_to_local_bbox(self, bbox: BoundingBox) -> BoundingBox:
        """Transform global bbox to tile-local coordinates."""
        x_min = bbox.x_min - self.x_offset
        y_min = bbox.y_min - self.y_offset
        x_max = bbox.x_max - self.x_offset
        y_max = bbox.y_max - self.y_offset

        if self.scale_factor != 1.0:
            x_min = int(x_min * self.scale_factor)
            y_min = int(y_min * self.scale_factor)
            x_max = int(x_max * self.scale_factor)
            y_max = int(y_max * self.scale_factor)

        return BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)

    def local_to_global_mask(self, mask: np.ndarray, global_shape: tuple[int, int]) -> np.ndarray:
        """Transform tile mask to global coordinates."""
        import cv2

        global_mask = np.zeros(global_shape, dtype=np.uint8)

        # Handle scaling
        if self.scale_factor != 1.0:
            orig_h = int(mask.shape[0] / self.scale_factor)
            orig_w = int(mask.shape[1] / self.scale_factor)
            mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        # Calculate placement region
        y_start = self.y_offset
        x_start = self.x_offset
        y_end = min(y_start + mask.shape[0], global_shape[0])
        x_end = min(x_start + mask.shape[1], global_shape[1])

        mask_h = y_end - y_start
        mask_w = x_end - x_start

        global_mask[y_start:y_end, x_start:x_end] = mask[:mask_h, :mask_w]

        return global_mask

    def local_point_to_global(self, x: int, y: int) -> tuple[int, int]:
        """Transform local point to global coordinates."""
        if self.scale_factor != 1.0:
            inv_scale = 1.0 / self.scale_factor
            x = int(x * inv_scale)
            y = int(y * inv_scale)
        return (x + self.x_offset, y + self.y_offset)

    def global_point_to_local(self, x: int, y: int) -> tuple[int, int]:
        """Transform global point to local coordinates."""
        local_x = x - self.x_offset
        local_y = y - self.y_offset
        if self.scale_factor != 1.0:
            local_x = int(local_x * self.scale_factor)
            local_y = int(local_y * self.scale_factor)
        return (local_x, local_y)


class ImageTiler:
    """Splits large images into overlapping tiles."""

    def __init__(self, config: TileConfig | None = None):
        self.config = config or TileConfig()

    def should_tile(self, image: np.ndarray) -> bool:
        """Check if image should be tiled."""
        h, w = image.shape[:2]
        tile_w, tile_h = self.config.tile_size
        return w > tile_w or h > tile_h

    def create_tiles(self, image: np.ndarray) -> list[TileInfo]:
        """
        Split image into overlapping tiles.

        Args:
            image: Input image (H, W, C)

        Returns:
            List of TileInfo objects
        """
        h, w = image.shape[:2]
        tile_w, tile_h = self.config.tile_size
        overlap = self.config.overlap

        tiles = []
        tile_id = 0

        # Calculate step size
        step_x = tile_w - overlap
        step_y = tile_h - overlap

        y = 0
        while y < h:
            x = 0
            while x < w:
                # Calculate tile bounds
                x_end = min(x + tile_w, w)
                y_end = min(y + tile_h, h)

                # Skip if tile is too small
                if (x_end - x) < self.config.min_tile_size or (y_end - y) < self.config.min_tile_size:
                    x += step_x
                    continue

                # Extract tile
                tile_image = image[y:y_end, x:x_end].copy()

                # Handle padding if needed
                if self.config.pad_tiles and (tile_image.shape[0] < tile_h or tile_image.shape[1] < tile_w):
                    padded = np.zeros((tile_h, tile_w, image.shape[2]), dtype=image.dtype)
                    padded[:tile_image.shape[0], :tile_image.shape[1]] = tile_image
                    tile_image = padded

                # Handle scaling if needed
                scale_factor = 1.0
                if self.config.scale_tiles:
                    import cv2
                    tile_image = cv2.resize(tile_image, (tile_w, tile_h))
                    scale_factor = tile_w / (x_end - x)

                tiles.append(TileInfo(
                    tile_id=tile_id,
                    image=tile_image,
                    x_offset=x,
                    y_offset=y,
                    original_width=x_end - x,
                    original_height=y_end - y,
                    scale_factor=scale_factor,
                ))

                tile_id += 1
                x += step_x

            y += step_y

        return tiles

    def stitch_masks(
        self,
        tile_masks: list[tuple[TileInfo, np.ndarray]],
        output_shape: tuple[int, int],
    ) -> np.ndarray:
        """
        Combine tile masks into single global mask using union.

        Args:
            tile_masks: List of (TileInfo, mask) tuples
            output_shape: (H, W) of output

        Returns:
            Combined mask
        """
        result = np.zeros(output_shape, dtype=np.uint8)

        for tile_info, mask in tile_masks:
            global_mask = tile_info.local_to_global_mask(mask, output_shape)
            result = np.maximum(result, global_mask)

        return result
