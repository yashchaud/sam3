"""
Temporal Tile Strategy for Video Processing.

This module implements a frame-cycling tile strategy where:
    - Frame 0: Process full image (catch large defects)
    - Frame 1-4: Process one tile each (catch fine defects)
    - Frame 4: Merge all detections from frames 0-4
    - Cycle repeats

This leverages the fact that video frames are largely static (20-30fps),
so we can spread tile processing across time without missing detections.

Key insight:
    - Full image pass catches large/obvious defects
    - Tile passes catch fine details that would be missed at full resolution
    - Temporal accumulation ensures nothing is missed
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
import numpy as np

from anomaly_detection.models.data_models import (
    Detection,
    BoundingBox,
    SegmentationMask,
)
from anomaly_detection.tiling.tiler import TileInfo, TileConfig


class FrameType(Enum):
    """Type of processing for current frame."""
    FULL_IMAGE = "full"      # Process entire image
    TILE = "tile"            # Process single tile
    CYCLE_COMPLETE = "cycle" # Tile processing + merge results


@dataclass
class FrameStrategy:
    """
    Strategy for processing current frame.

    Attributes:
        frame_type: Whether to process full image or tile
        tile_index: Which tile to process (0-3 for 2x2 grid), None for full
        tile_info: Tile metadata if processing a tile
        is_cycle_end: True if this frame completes the cycle (time to merge)
    """
    frame_type: FrameType
    tile_index: Optional[int] = None
    tile_info: Optional[TileInfo] = None
    is_cycle_end: bool = False


@dataclass
class AccumulatedDetection:
    """
    Detection accumulated across frame cycle.

    Tracks both the original detection and its source context
    for proper coordinate handling.
    """
    detection: Detection
    source_frame: int          # Which frame in cycle (0-4)
    tile_info: Optional[TileInfo]  # None if from full image pass
    global_bbox: BoundingBox   # Always in global coordinates

    # Segmentation results (populated when SAM runs)
    local_mask: Optional[np.ndarray] = None
    global_mask: Optional[np.ndarray] = None
    sam_score: float = 0.0


@dataclass
class CycleResult:
    """
    Result of a complete processing cycle (frames 0-4).

    Contains all merged detections with global coordinates.
    """
    anomalies: list[AccumulatedDetection]
    structures: list[AccumulatedDetection]
    cycle_number: int
    frame_count: int  # Usually 5 (1 full + 4 tiles)


@dataclass
class TemporalTileConfig:
    """
    Configuration for temporal tile strategy.

    Attributes:
        grid_size: Tile grid dimensions (rows, cols). Default (2, 2) = 4 tiles
        tile_scale: Scale factor for tiles (1.0 = original crop size)
        full_image_scale: Scale for full image pass (< 1.0 to fit in memory)
        overlap: Overlap between tiles in pixels
        cycle_length: Total frames per cycle (1 full + N tiles)
    """
    grid_size: tuple[int, int] = (2, 2)  # 2x2 = 4 tiles
    tile_scale: float = 1.0  # Process tiles at full resolution
    full_image_scale: float = 1.0  # May need to reduce for large images
    overlap: int = 64  # Tile overlap in pixels

    @property
    def num_tiles(self) -> int:
        return self.grid_size[0] * self.grid_size[1]

    @property
    def cycle_length(self) -> int:
        """Total frames in one cycle: 1 (full) + num_tiles."""
        return 1 + self.num_tiles


class TemporalTileManager:
    """
    Manages temporal tile cycling for video processing.

    Implements the frame cycling strategy:
        Frame 0: Full image detection
        Frame 1: Tile 0 (top-left)
        Frame 2: Tile 1 (top-right)
        Frame 3: Tile 2 (bottom-left)
        Frame 4: Tile 3 (bottom-right) + MERGE

    Usage:
        manager = TemporalTileManager(config)

        for frame_idx, frame in enumerate(video_frames):
            # Get strategy for this frame
            strategy = manager.get_frame_strategy(frame_idx, frame)

            if strategy.frame_type == FrameType.FULL_IMAGE:
                # Process full image
                image_to_process = manager.prepare_full_image(frame)
                detections = detector.detect(image_to_process)
                manager.accumulate_full_image(detections, frame)

            else:
                # Process tile
                tile_image = strategy.tile_info.tile_image
                detections = detector.detect(tile_image)
                # SAM uses tile_image + tile-local bbox!
                manager.accumulate_tile(detections, strategy.tile_info)

            if strategy.is_cycle_end:
                # Merge and output
                result = manager.complete_cycle()
                yield result
    """

    def __init__(self, config: Optional[TemporalTileConfig] = None):
        """
        Initialize the temporal tile manager.

        Args:
            config: Temporal tiling configuration
        """
        self.config = config or TemporalTileConfig()

        # Cycle state
        self._frame_in_cycle = 0  # 0 to cycle_length-1
        self._cycle_number = 0
        self._current_image_shape: Optional[tuple[int, int]] = None

        # Accumulated detections for current cycle
        self._accumulated_anomalies: list[AccumulatedDetection] = []
        self._accumulated_structures: list[AccumulatedDetection] = []

        # Pre-computed tile infos (computed once per image size)
        self._tile_infos: list[TileInfo] = []

    def reset(self) -> None:
        """Reset state for new video."""
        self._frame_in_cycle = 0
        self._cycle_number = 0
        self._current_image_shape = None
        self._accumulated_anomalies = []
        self._accumulated_structures = []
        self._tile_infos = []

    def get_frame_strategy(
        self,
        frame_index: int,
        image: np.ndarray,
    ) -> FrameStrategy:
        """
        Determine processing strategy for current frame.

        Args:
            frame_index: Global frame index in video
            image: Current frame image

        Returns:
            FrameStrategy indicating how to process this frame
        """
        h, w = image.shape[:2]

        # Recompute tiles if image size changed
        if self._current_image_shape != (h, w):
            self._current_image_shape = (h, w)
            self._tile_infos = self._compute_tile_infos(image)

        # Determine position in cycle
        cycle_pos = frame_index % self.config.cycle_length
        self._frame_in_cycle = cycle_pos

        if cycle_pos == 0:
            # Full image frame
            return FrameStrategy(
                frame_type=FrameType.FULL_IMAGE,
                tile_index=None,
                tile_info=None,
                is_cycle_end=False,
            )
        else:
            # Tile frame
            tile_idx = cycle_pos - 1  # 0-indexed tile
            tile_info = self._tile_infos[tile_idx]

            # Extract tile from current frame
            tile_info = self._extract_tile(image, tile_idx)

            is_last_tile = (cycle_pos == self.config.cycle_length - 1)

            return FrameStrategy(
                frame_type=FrameType.TILE if not is_last_tile else FrameType.CYCLE_COMPLETE,
                tile_index=tile_idx,
                tile_info=tile_info,
                is_cycle_end=is_last_tile,
            )

    def _compute_tile_infos(self, image: np.ndarray) -> list[TileInfo]:
        """Pre-compute tile metadata for current image size."""
        h, w = image.shape[:2]
        rows, cols = self.config.grid_size
        overlap = self.config.overlap

        # Tile dimensions
        tile_h = (h + overlap * (rows - 1)) // rows
        tile_w = (w + overlap * (cols - 1)) // cols

        tiles = []
        tile_idx = 0

        for row in range(rows):
            for col in range(cols):
                # Calculate tile position
                y = row * (tile_h - overlap)
                x = col * (tile_w - overlap)

                # Clamp to image bounds
                y_end = min(y + tile_h, h)
                x_end = min(x + tile_w, w)
                actual_h = y_end - y
                actual_w = x_end - x

                # Scale factor (if tiles are scaled)
                scale = self.config.tile_scale

                tile_info = TileInfo(
                    tile_id=f"tile_{tile_idx:02d}",
                    row=row,
                    col=col,
                    global_x=x,
                    global_y=y,
                    crop_width=actual_w,
                    crop_height=actual_h,
                    tile_width=int(actual_w * scale),
                    tile_height=int(actual_h * scale),
                    scale_x=scale,
                    scale_y=scale,
                    tile_image=None,  # Will be set when extracting
                )
                tiles.append(tile_info)
                tile_idx += 1

        return tiles

    def _extract_tile(self, image: np.ndarray, tile_idx: int) -> TileInfo:
        """Extract tile image and return updated TileInfo."""
        import cv2

        base_info = self._tile_infos[tile_idx]

        # Extract crop from image
        y = base_info.global_y
        x = base_info.global_x
        crop = image[y:y+base_info.crop_height, x:x+base_info.crop_width].copy()

        # Scale if needed
        if self.config.tile_scale != 1.0:
            tile_image = cv2.resize(
                crop,
                (base_info.tile_width, base_info.tile_height),
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            tile_image = crop

        # Return new TileInfo with image attached
        return TileInfo(
            tile_id=base_info.tile_id,
            row=base_info.row,
            col=base_info.col,
            global_x=base_info.global_x,
            global_y=base_info.global_y,
            crop_width=base_info.crop_width,
            crop_height=base_info.crop_height,
            tile_width=base_info.tile_width,
            tile_height=base_info.tile_height,
            scale_x=base_info.scale_x,
            scale_y=base_info.scale_y,
            tile_image=tile_image,
        )

    def prepare_full_image(self, image: np.ndarray) -> np.ndarray:
        """
        Prepare full image for detection.

        May scale down if configured for memory efficiency.
        """
        import cv2

        if self.config.full_image_scale == 1.0:
            return image

        h, w = image.shape[:2]
        new_h = int(h * self.config.full_image_scale)
        new_w = int(w * self.config.full_image_scale)

        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    def accumulate_full_image(
        self,
        anomalies: list[Detection],
        structures: list[Detection],
        image: np.ndarray,
    ) -> None:
        """
        Accumulate detections from full image pass.

        For full image, coordinates are already global (possibly scaled).
        """
        scale = self.config.full_image_scale

        for det in anomalies:
            # Unscale bbox if full image was scaled
            if scale != 1.0:
                global_bbox = BoundingBox(
                    x_min=det.bbox.x_min / scale,
                    y_min=det.bbox.y_min / scale,
                    x_max=det.bbox.x_max / scale,
                    y_max=det.bbox.y_max / scale,
                )
            else:
                global_bbox = det.bbox

            acc_det = AccumulatedDetection(
                detection=det,
                source_frame=0,
                tile_info=None,  # Full image, no tile
                global_bbox=global_bbox,
            )
            self._accumulated_anomalies.append(acc_det)

        for det in structures:
            if scale != 1.0:
                global_bbox = BoundingBox(
                    x_min=det.bbox.x_min / scale,
                    y_min=det.bbox.y_min / scale,
                    x_max=det.bbox.x_max / scale,
                    y_max=det.bbox.y_max / scale,
                )
            else:
                global_bbox = det.bbox

            acc_det = AccumulatedDetection(
                detection=det,
                source_frame=0,
                tile_info=None,
                global_bbox=global_bbox,
            )
            self._accumulated_structures.append(acc_det)

    def accumulate_tile(
        self,
        anomalies: list[Detection],
        structures: list[Detection],
        tile_info: TileInfo,
    ) -> None:
        """
        Accumulate detections from tile pass.

        IMPORTANT: Detection bboxes are in TILE-LOCAL coordinates.
        We store both for later SAM processing:
            - detection.bbox = tile-local (for SAM)
            - global_bbox = transformed to global (for merging/output)
        """
        frame_in_cycle = self._frame_in_cycle

        for det in anomalies:
            # Transform to global coordinates
            gx_min, gy_min, gx_max, gy_max = tile_info.local_to_global_bbox(
                det.bbox.x_min,
                det.bbox.y_min,
                det.bbox.x_max,
                det.bbox.y_max,
            )
            global_bbox = BoundingBox(
                x_min=gx_min, y_min=gy_min,
                x_max=gx_max, y_max=gy_max,
            )

            acc_det = AccumulatedDetection(
                detection=det,  # Keeps tile-local bbox for SAM!
                source_frame=frame_in_cycle,
                tile_info=tile_info,  # Keep tile info for SAM!
                global_bbox=global_bbox,
            )
            self._accumulated_anomalies.append(acc_det)

        for det in structures:
            gx_min, gy_min, gx_max, gy_max = tile_info.local_to_global_bbox(
                det.bbox.x_min,
                det.bbox.y_min,
                det.bbox.x_max,
                det.bbox.y_max,
            )
            global_bbox = BoundingBox(
                x_min=gx_min, y_min=gy_min,
                x_max=gx_max, y_max=gy_max,
            )

            acc_det = AccumulatedDetection(
                detection=det,
                source_frame=frame_in_cycle,
                tile_info=tile_info,
                global_bbox=global_bbox,
            )
            self._accumulated_structures.append(acc_det)

    def add_segmentation(
        self,
        detection_id: str,
        local_mask: np.ndarray,
        global_mask: np.ndarray,
        sam_score: float,
    ) -> None:
        """
        Add segmentation result to accumulated detection.

        Call this after running SAM on each detection.
        """
        for acc_det in self._accumulated_anomalies:
            if acc_det.detection.detection_id == detection_id:
                acc_det.local_mask = local_mask
                acc_det.global_mask = global_mask
                acc_det.sam_score = sam_score
                return

    def complete_cycle(self) -> CycleResult:
        """
        Complete current cycle and return merged results.

        Applies NMS to remove duplicate detections from overlapping regions.
        Resets accumulator for next cycle.
        """
        # Deduplicate detections
        deduped_anomalies = self._deduplicate(self._accumulated_anomalies)
        deduped_structures = self._deduplicate(self._accumulated_structures)

        result = CycleResult(
            anomalies=deduped_anomalies,
            structures=deduped_structures,
            cycle_number=self._cycle_number,
            frame_count=self.config.cycle_length,
        )

        # Reset for next cycle
        self._accumulated_anomalies = []
        self._accumulated_structures = []
        self._cycle_number += 1

        return result

    def _deduplicate(
        self,
        detections: list[AccumulatedDetection],
        iou_threshold: float = 0.5,
    ) -> list[AccumulatedDetection]:
        """
        Remove duplicate detections using NMS on global bboxes.

        Prefers detections from tile passes over full image (higher resolution).
        """
        if len(detections) <= 1:
            return detections

        # Sort by: tile detections first (higher res), then by confidence
        sorted_dets = sorted(
            detections,
            key=lambda d: (
                0 if d.tile_info is not None else 1,  # Tiles first
                -d.detection.confidence,  # Higher confidence first
            ),
        )

        kept = []
        suppressed = set()

        for i, det in enumerate(sorted_dets):
            if i in suppressed:
                continue

            kept.append(det)

            # Suppress overlapping lower-priority detections
            for j in range(i + 1, len(sorted_dets)):
                if j in suppressed:
                    continue

                iou = self._compute_iou(det.global_bbox, sorted_dets[j].global_bbox)
                if iou > iou_threshold:
                    suppressed.add(j)

        return kept

    def _compute_iou(self, box1: BoundingBox, box2: BoundingBox) -> float:
        """Compute IoU between two bounding boxes."""
        x_min = max(box1.x_min, box2.x_min)
        y_min = max(box1.y_min, box2.y_min)
        x_max = min(box1.x_max, box2.x_max)
        y_max = min(box1.y_max, box2.y_max)

        if x_max <= x_min or y_max <= y_min:
            return 0.0

        intersection = (x_max - x_min) * (y_max - y_min)
        area1 = box1.area
        area2 = box2.area
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    @property
    def current_cycle(self) -> int:
        """Current cycle number."""
        return self._cycle_number

    @property
    def frame_in_cycle(self) -> int:
        """Current frame position within cycle (0 to cycle_length-1)."""
        return self._frame_in_cycle

    @property
    def pending_anomalies(self) -> list[AccumulatedDetection]:
        """Anomalies accumulated so far in current cycle."""
        return self._accumulated_anomalies.copy()

    @property
    def pending_structures(self) -> list[AccumulatedDetection]:
        """Structures accumulated so far in current cycle."""
        return self._accumulated_structures.copy()
