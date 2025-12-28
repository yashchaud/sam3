"""
Tiled Detection Coordinator.

Orchestrates the detection and segmentation pipeline for tiled images,
ensuring correct coordinate handling at each stage.

CRITICAL INSIGHT:
    When processing tiles, SAM must receive:
    1. The TILE IMAGE (not the global image)
    2. TILE-LOCAL bounding box coordinates (not global)

    Only after SAM produces the mask do we transform both the
    bounding box and mask to global coordinates.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np

from anomaly_detection.models.data_models import (
    Detection,
    DetectionType,
    BoundingBox,
    SegmentationMask,
    generate_id,
)
from anomaly_detection.tiling.tiler import (
    TileInfo,
    TileConfig,
    ImageTiler,
)


@dataclass
class TiledDetection:
    """
    A detection with both tile-local and global coordinates.

    Attributes:
        detection: Original detection with tile-local bbox
        tile_info: Tile metadata for coordinate transforms
        global_bbox: Bounding box in global image coordinates
    """
    detection: Detection
    tile_info: TileInfo
    global_bbox: BoundingBox


@dataclass
class TiledSegmentationResult:
    """
    Segmentation result with proper coordinate handling.

    Attributes:
        detection: The original detection
        tile_info: Tile this detection came from
        local_mask: Mask in tile coordinates
        global_mask: Mask transformed to global coordinates
        global_bbox: Bounding box in global coordinates
        sam_score: Segmentation confidence score
    """
    detection: Detection
    tile_info: TileInfo
    local_mask: np.ndarray
    global_mask: np.ndarray
    global_bbox: BoundingBox
    sam_score: float


def compute_iou(box1: BoundingBox, box2: BoundingBox) -> float:
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


class TiledDetectionCoordinator:
    """
    Coordinates tiled detection and segmentation with correct coordinate handling.

    This class solves the coordinate transformation problem:

    1. DETECTION PHASE:
       - Split image into tiles
       - Run RF-DETR on each tile -> tile-local bounding boxes
       - Transform bboxes to global coordinates for deduplication

    2. DEDUPLICATION:
       - Apply NMS on global bboxes to remove duplicates from overlaps

    3. SEGMENTATION PHASE (CRITICAL):
       - For each detection, pass to SAM:
         * The TILE IMAGE (tile_info.tile_image)
         * The TILE-LOCAL bbox (detection.bbox, NOT global)
       - SAM produces tile-local mask

    4. MASK TRANSFORMATION:
       - Transform local masks to global coordinates
       - Handle scaling (inverse of tile scaling)
       - Handle translation (add tile offset)

    Usage:
        coordinator = TiledDetectionCoordinator(
            detector=rf_detr_detector,
            segmenter=sam_segmenter,
            tile_config=TileConfig(tile_size=(640, 640)),
        )

        results = coordinator.process(large_image)

        for result in results:
            # result.global_bbox - bounding box in original image coords
            # result.global_mask - mask in original image coords
            print(f"Found {result.detection.class_name} at {result.global_bbox}")
    """

    def __init__(
        self,
        detector,  # RFDETRDetector
        segmenter,  # SAM3Segmenter
        tile_config: Optional[TileConfig] = None,
        nms_iou_threshold: float = 0.5,
    ):
        """
        Initialize the coordinator.

        Args:
            detector: RF-DETR detector instance (must be loaded)
            segmenter: SAM segmenter instance (must be loaded)
            tile_config: Tiling configuration
            nms_iou_threshold: IoU threshold for NMS deduplication
        """
        self.detector = detector
        self.segmenter = segmenter
        self.tile_config = tile_config or TileConfig()
        self.nms_iou_threshold = nms_iou_threshold
        self._tiler = ImageTiler(self.tile_config)

    def process(
        self,
        image: np.ndarray,
        segment_structures: bool = False,
    ) -> tuple[list[TiledSegmentationResult], list[TiledDetection]]:
        """
        Process image with tiled detection and segmentation.

        Args:
            image: Input image (H, W, C) in RGB format
            segment_structures: Whether to segment structural elements too

        Returns:
            Tuple of:
                - List of segmentation results (anomalies, optionally structures)
                - List of structure detections (without segmentation if not requested)
        """
        global_h, global_w = image.shape[:2]
        global_shape = (global_h, global_w)

        # Step 1: Create tiles
        tiles = self._tiler.create_tiles(image)

        # Step 2: Run detection on all tiles
        all_detections: list[TiledDetection] = []
        structure_detections: list[TiledDetection] = []

        for tile_info in tiles:
            # Detect on tile image
            detector_output = self.detector.detect(tile_info.tile_image)

            # Process anomaly detections
            for det in detector_output.anomalies:
                tiled_det = self._create_tiled_detection(det, tile_info)
                all_detections.append(tiled_det)

            # Process structure detections
            for det in detector_output.structures:
                tiled_det = self._create_tiled_detection(det, tile_info)
                structure_detections.append(tiled_det)

        # Step 3: Deduplicate detections across tile overlaps
        anomaly_detections = self._nms_deduplicate(all_detections)
        structure_detections = self._nms_deduplicate(structure_detections)

        # Step 4: Segment anomalies (and optionally structures)
        segmentation_results = []

        detections_to_segment = anomaly_detections
        if segment_structures:
            detections_to_segment = anomaly_detections + structure_detections

        for tiled_det in detections_to_segment:
            result = self._segment_detection(tiled_det, global_shape)
            if result is not None:
                segmentation_results.append(result)

        return segmentation_results, structure_detections

    def process_anomalies_only(
        self,
        image: np.ndarray,
    ) -> tuple[list[TiledSegmentationResult], list[Detection]]:
        """
        Process image, segmenting only anomalies.

        More efficient than process() when you don't need structure masks.

        Args:
            image: Input image (H, W, C)

        Returns:
            Tuple of (anomaly segmentation results, structure detections with global coords)
        """
        seg_results, struct_tiled = self.process(image, segment_structures=False)

        # Convert structure detections to regular Detections with global bbox
        structures = []
        for tiled_det in struct_tiled:
            # Create new detection with global bbox
            struct = Detection(
                detection_id=tiled_det.detection.detection_id,
                detection_type=tiled_det.detection.detection_type,
                class_name=tiled_det.detection.class_name,
                confidence=tiled_det.detection.confidence,
                bbox=tiled_det.global_bbox,
            )
            structures.append(struct)

        return seg_results, structures

    def _create_tiled_detection(
        self,
        detection: Detection,
        tile_info: TileInfo,
    ) -> TiledDetection:
        """Create TiledDetection with global bbox computed."""
        # Transform tile-local bbox to global
        gx_min, gy_min, gx_max, gy_max = tile_info.local_to_global_bbox(
            detection.bbox.x_min,
            detection.bbox.y_min,
            detection.bbox.x_max,
            detection.bbox.y_max,
        )

        global_bbox = BoundingBox(
            x_min=gx_min,
            y_min=gy_min,
            x_max=gx_max,
            y_max=gy_max,
        )

        return TiledDetection(
            detection=detection,
            tile_info=tile_info,
            global_bbox=global_bbox,
        )

    def _nms_deduplicate(
        self,
        detections: list[TiledDetection],
    ) -> list[TiledDetection]:
        """
        Apply NMS to remove duplicate detections from tile overlaps.

        Uses global bounding boxes for IoU computation.
        Groups by class for class-aware NMS.

        Args:
            detections: List of tiled detections

        Returns:
            Deduplicated list of detections
        """
        if len(detections) <= 1:
            return detections

        # Group by class
        by_class: dict[str, list[TiledDetection]] = {}
        for det in detections:
            class_name = det.detection.class_name
            if class_name not in by_class:
                by_class[class_name] = []
            by_class[class_name].append(det)

        # Apply NMS per class
        kept = []
        for class_name, class_dets in by_class.items():
            kept.extend(self._nms_single_class(class_dets))

        return kept

    def _nms_single_class(
        self,
        detections: list[TiledDetection],
    ) -> list[TiledDetection]:
        """Apply NMS to detections of a single class."""
        if len(detections) <= 1:
            return detections

        # Sort by confidence (highest first)
        sorted_dets = sorted(
            detections,
            key=lambda d: d.detection.confidence,
            reverse=True,
        )

        kept = []
        suppressed = set()

        for i, det in enumerate(sorted_dets):
            if i in suppressed:
                continue

            kept.append(det)

            # Suppress lower-confidence overlapping detections
            for j in range(i + 1, len(sorted_dets)):
                if j in suppressed:
                    continue

                iou = compute_iou(det.global_bbox, sorted_dets[j].global_bbox)
                if iou > self.nms_iou_threshold:
                    suppressed.add(j)

        return kept

    def _segment_detection(
        self,
        tiled_det: TiledDetection,
        global_shape: tuple[int, int],
    ) -> Optional[TiledSegmentationResult]:
        """
        Segment a detection using SAM with correct coordinate handling.

        CRITICAL: SAM receives:
            - tile_info.tile_image (the tile, not global image)
            - detection.bbox (tile-local coordinates)

        The resulting mask is then transformed to global coordinates.

        Args:
            tiled_det: Detection with tile info
            global_shape: (height, width) of original image

        Returns:
            TiledSegmentationResult or None if segmentation failed
        """
        tile_info = tiled_det.tile_info
        detection = tiled_det.detection

        # Set the TILE image in SAM (not the global image!)
        self.segmenter.set_image(tile_info.tile_image)

        try:
            # Segment using TILE-LOCAL bbox
            seg_result = self.segmenter.segment_detection(detection)

            if not seg_result.success:
                return None

            local_mask = seg_result.mask.mask

            # Transform mask to global coordinates
            global_mask = tile_info.local_to_global_mask(local_mask, global_shape)

            return TiledSegmentationResult(
                detection=detection,
                tile_info=tile_info,
                local_mask=local_mask,
                global_mask=global_mask,
                global_bbox=tiled_det.global_bbox,
                sam_score=seg_result.mask.sam_score,
            )

        except Exception as e:
            # Log error but continue processing
            print(f"Segmentation failed for {detection.detection_id}: {e}")
            return None

        finally:
            # Clear image embedding
            self.segmenter.clear_image()

    def detect_only(self, image: np.ndarray) -> tuple[list[TiledDetection], list[TiledDetection]]:
        """
        Run only detection phase (no segmentation).

        Useful for quick preview or when masks aren't needed.

        Args:
            image: Input image (H, W, C)

        Returns:
            Tuple of (anomaly detections, structure detections) with global coords
        """
        tiles = self._tiler.create_tiles(image)

        anomaly_detections: list[TiledDetection] = []
        structure_detections: list[TiledDetection] = []

        for tile_info in tiles:
            detector_output = self.detector.detect(tile_info.tile_image)

            for det in detector_output.anomalies:
                tiled_det = self._create_tiled_detection(det, tile_info)
                anomaly_detections.append(tiled_det)

            for det in detector_output.structures:
                tiled_det = self._create_tiled_detection(det, tile_info)
                structure_detections.append(tiled_det)

        # Deduplicate
        anomaly_detections = self._nms_deduplicate(anomaly_detections)
        structure_detections = self._nms_deduplicate(structure_detections)

        return anomaly_detections, structure_detections
