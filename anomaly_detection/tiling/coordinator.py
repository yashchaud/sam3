"""Tiled detection coordinator for large images."""

from dataclasses import dataclass
import numpy as np

from .tiler import ImageTiler, TileInfo, TileConfig
from ..models import Detection, BoundingBox, SegmentationMask
from ..detector import RFDETRDetector
from ..segmenter import SAM3Segmenter


@dataclass
class TiledSegmentationResult:
    """Result from tiled segmentation."""
    detection: Detection
    tile_info: TileInfo
    local_mask: np.ndarray | None
    global_mask: np.ndarray | None
    global_bbox: BoundingBox
    sam_score: float | None


@dataclass
class TiledDetection:
    """Detection with tile information."""
    detection: Detection
    tile_info: TileInfo
    global_bbox: BoundingBox


class TiledDetectionCoordinator:
    """
    Coordinates detection and segmentation across tiled images.

    Critical coordinate handling:
    1. RF-DETR receives TILE IMAGE -> tile-local bboxes
    2. Transform to GLOBAL coords for NMS deduplication
    3. For SAM: pass TILE IMAGE + TILE-LOCAL bbox
    4. After SAM: transform mask and bbox to GLOBAL coords
    """

    def __init__(
        self,
        detector: RFDETRDetector,
        segmenter: SAM3Segmenter,
        tile_config: TileConfig | None = None,
        nms_iou_threshold: float = 0.5,
    ):
        self.detector = detector
        self.segmenter = segmenter
        self.tiler = ImageTiler(tile_config)
        self.nms_iou_threshold = nms_iou_threshold

    def process(
        self,
        image: np.ndarray,
        segment_structures: bool = False,
    ) -> tuple[list[TiledSegmentationResult], list[Detection]]:
        """
        Full detection + segmentation on tiled image.

        Args:
            image: Full image (H, W, C)
            segment_structures: Whether to segment structures too

        Returns:
            (segmentation_results, structure_detections)
        """
        h, w = image.shape[:2]

        # Create tiles
        tiles = self.tiler.create_tiles(image)

        # Collect all detections
        all_detections: list[TiledDetection] = []

        for tile in tiles:
            # Run detection on tile
            output = self.detector.detect(tile.image)

            for detection in output.detections:
                # Transform bbox to global
                global_bbox = tile.local_to_global_bbox(detection.bbox)

                all_detections.append(TiledDetection(
                    detection=detection,
                    tile_info=tile,
                    global_bbox=global_bbox,
                ))

        # NMS deduplication using global bboxes
        filtered = self._nms_detections(all_detections)

        # Separate by type
        anomalies = [d for d in filtered if d.detection.detection_type.value == "anomaly"]
        structures = [d for d in filtered if d.detection.detection_type.value == "structure"]

        # Segment anomalies (and optionally structures)
        to_segment = anomalies + (structures if segment_structures else [])
        seg_results = self._segment_tiled(to_segment, image.shape[:2])

        # Return structure detections in global coords
        structure_dets = [
            Detection(
                detection_id=d.detection.detection_id,
                detection_type=d.detection.detection_type,
                class_name=d.detection.class_name,
                confidence=d.detection.confidence,
                bbox=d.global_bbox,
            )
            for d in structures
        ]

        return seg_results, structure_dets

    def process_anomalies_only(
        self,
        image: np.ndarray,
    ) -> tuple[list[TiledSegmentationResult], list[Detection]]:
        """Process only anomalies, faster variant."""
        return self.process(image, segment_structures=False)

    def detect_only(self, image: np.ndarray) -> list[TiledDetection]:
        """Run detection only without segmentation."""
        tiles = self.tiler.create_tiles(image)

        all_detections: list[TiledDetection] = []

        for tile in tiles:
            output = self.detector.detect(tile.image)

            for detection in output.detections:
                global_bbox = tile.local_to_global_bbox(detection.bbox)

                all_detections.append(TiledDetection(
                    detection=detection,
                    tile_info=tile,
                    global_bbox=global_bbox,
                ))

        return self._nms_detections(all_detections)

    def _segment_tiled(
        self,
        tiled_detections: list[TiledDetection],
        global_shape: tuple[int, int],
    ) -> list[TiledSegmentationResult]:
        """Segment detections using their original tiles."""
        results = []

        # Group by tile for efficient processing
        by_tile: dict[int, list[TiledDetection]] = {}
        for td in tiled_detections:
            tile_id = td.tile_info.tile_id
            if tile_id not in by_tile:
                by_tile[tile_id] = []
            by_tile[tile_id].append(td)

        for tile_id, detections in by_tile.items():
            if not detections:
                continue

            tile_info = detections[0].tile_info

            # Set tile image for batch processing
            self.segmenter.set_image(tile_info.image)

            try:
                for td in detections:
                    # Segment using TILE-LOCAL bbox (critical!)
                    seg_result = self.segmenter.segment_detection(td.detection)

                    if seg_result.success and seg_result.mask is not None:
                        # Transform mask to global
                        global_mask = tile_info.local_to_global_mask(
                            seg_result.mask.data,
                            global_shape,
                        )

                        results.append(TiledSegmentationResult(
                            detection=td.detection,
                            tile_info=tile_info,
                            local_mask=seg_result.mask.data,
                            global_mask=global_mask,
                            global_bbox=td.global_bbox,
                            sam_score=seg_result.mask.sam_score,
                        ))
                    else:
                        results.append(TiledSegmentationResult(
                            detection=td.detection,
                            tile_info=tile_info,
                            local_mask=None,
                            global_mask=None,
                            global_bbox=td.global_bbox,
                            sam_score=None,
                        ))
            finally:
                self.segmenter.clear_image()

        return results

    def _nms_detections(
        self,
        detections: list[TiledDetection],
    ) -> list[TiledDetection]:
        """Apply NMS using global bboxes."""
        if not detections:
            return []

        # Sort by confidence
        sorted_dets = sorted(detections, key=lambda d: d.detection.confidence, reverse=True)

        keep = []
        for det in sorted_dets:
            should_keep = True

            for kept in keep:
                # Compare same-class detections
                if det.detection.class_name == kept.detection.class_name:
                    iou = det.global_bbox.iou(kept.global_bbox)
                    if iou > self.nms_iou_threshold:
                        should_keep = False
                        break

            if should_keep:
                keep.append(det)

        return keep
