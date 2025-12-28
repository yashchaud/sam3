"""
Video Processor with Temporal Tiling.

Processes video frames using a temporal tile strategy:
    - Frame 0: Full image pass (catches large defects)
    - Frames 1-4: One tile per frame (catches fine details)
    - Every 5 frames: Merge all detections

This approach leverages temporal redundancy in video (frames are similar)
to achieve high-resolution detection without processing overhead.

Critical Coordinate Handling:
    - Full image detections: Already in global coordinates (may need unscaling)
    - Tile detections: SAM receives TILE IMAGE + TILE-LOCAL bbox
    - All outputs use GLOBAL coordinates
"""

from dataclasses import dataclass
from typing import Optional, Iterator, Generator
from pathlib import Path
from datetime import datetime
import numpy as np
import time

from anomaly_detection.models.data_models import (
    Detection,
    DetectionType,
    BoundingBox,
    SegmentationMask,
    AnomalyResult,
    PipelineOutput,
    generate_id,
)
from anomaly_detection.detector.rf_detr_detector import (
    RFDETRDetector,
    DetectorConfig,
    RFDETRVariant,
)
from anomaly_detection.segmenter.sam3_segmenter import (
    SAM3Segmenter,
    SegmenterConfig,
)
from anomaly_detection.association.structure_defect_matcher import (
    StructureDefectMatcher,
    MatcherConfig,
)
from anomaly_detection.geometry.mask_geometry import (
    MaskGeometryExtractor,
    GeometryConfig,
)
from anomaly_detection.tiling.temporal_tiler import (
    TemporalTileManager,
    TemporalTileConfig,
    FrameType,
    FrameStrategy,
    AccumulatedDetection,
    CycleResult,
)


@dataclass
class VideoConfig:
    """
    Configuration for video processing.

    Attributes:
        segmenter_model_path: Path to SAM3 weights
        detector_variant: RF-DETR variant
        detector_weights: Custom detector weights (optional)
        detector_num_classes: Number of classes if using custom weights
        device: Compute device

        tile_config: Temporal tiling configuration
        confidence_threshold: Detection confidence threshold

        save_masks: Whether to save masks to disk
        mask_output_dir: Directory for saved masks
    """
    segmenter_model_path: Path
    detector_variant: RFDETRVariant = RFDETRVariant.MEDIUM
    detector_weights: Optional[Path] = None
    detector_num_classes: Optional[int] = None
    device: str = "auto"

    tile_config: Optional[TemporalTileConfig] = None
    confidence_threshold: float = 0.3

    save_masks: bool = False
    mask_output_dir: Optional[Path] = None


@dataclass
class CycleOutput:
    """
    Output from processing one complete cycle (5 frames).

    Similar to PipelineOutput but specific to video processing.
    """
    cycle_id: str
    cycle_number: int
    start_frame: int
    end_frame: int
    image_width: int
    image_height: int
    timestamp: datetime

    anomalies: list[AnomalyResult]
    structures: list[Detection]

    total_time_ms: float
    detector_time_ms: float
    segmenter_time_ms: float


class VideoProcessor:
    """
    Video processor with temporal tile strategy.

    Processing flow for each 5-frame cycle:

    Frame 0 (Full Image):
        1. Optionally scale down for memory
        2. Run RF-DETR → global detections
        3. Accumulate detections

    Frames 1-4 (Tiles):
        1. Extract tile from frame
        2. Run RF-DETR on tile → tile-local detections
        3. Accumulate with tile metadata (for SAM later)

    End of Cycle (Frame 4):
        4. Deduplicate all accumulated detections (NMS)
        5. For each unique detection:
           - If from tile: SAM(tile_image, tile-local bbox) → tile-local mask
           - If from full: SAM(scaled_image, scaled bbox) → scaled mask
        6. Transform all masks to global coordinates
        7. Run association (anomaly ↔ structure)
        8. Extract geometry
        9. Output CycleResult

    Usage:
        processor = VideoProcessor(config)
        processor.load()

        for cycle_output in processor.process_video(video_frames):
            # cycle_output contains merged results every 5 frames
            for anomaly in cycle_output.anomalies:
                print(f"{anomaly.defect_type} at {anomaly.bbox}")

    Note:
        - Results are yielded every 5 frames (configurable via tile_config)
        - All coordinates in output are in original image space
    """

    def __init__(self, config: VideoConfig):
        """
        Initialize the video processor.

        Args:
            config: Video processing configuration
        """
        self.config = config

        # Build component configs
        detector_config = DetectorConfig(
            variant=config.detector_variant,
            pretrain_weights=config.detector_weights,
            num_classes=config.detector_num_classes,
            device=config.device,
            confidence_threshold=config.confidence_threshold,
        )

        segmenter_config = SegmenterConfig(
            model_path=config.segmenter_model_path,
            device=config.device,
        )

        # Initialize components
        self._detector = RFDETRDetector(detector_config)
        self._segmenter = SAM3Segmenter(segmenter_config)
        self._matcher = StructureDefectMatcher(MatcherConfig())
        self._geometry = MaskGeometryExtractor(GeometryConfig())

        # Temporal tile manager
        tile_config = config.tile_config or TemporalTileConfig()
        self._tile_manager = TemporalTileManager(tile_config)

        self._loaded = False
        self._current_full_image: Optional[np.ndarray] = None
        self._current_image_shape: Optional[tuple[int, int]] = None

    def load(self) -> None:
        """Load all models into memory."""
        self._detector.load()
        self._segmenter.load()
        self._loaded = True

    def unload(self) -> None:
        """Release models from memory."""
        self._detector.unload()
        self._segmenter.unload()
        self._tile_manager.reset()
        self._loaded = False

    def is_loaded(self) -> bool:
        """Check if processor is ready."""
        return self._loaded

    def process_video(
        self,
        frames: Iterator[np.ndarray],
        start_frame: int = 0,
    ) -> Generator[CycleOutput, None, None]:
        """
        Process video frames with temporal tiling.

        Yields a CycleOutput every cycle_length frames (default 5).

        Args:
            frames: Iterator of video frames (H, W, C) in RGB
            start_frame: Starting frame number for metadata

        Yields:
            CycleOutput for each complete processing cycle
        """
        if not self._loaded:
            raise RuntimeError("Processor not loaded. Call load() first.")

        self._tile_manager.reset()

        for frame_idx, frame in enumerate(frames):
            global_frame_idx = start_frame + frame_idx

            # Get strategy for this frame
            strategy = self._tile_manager.get_frame_strategy(frame_idx, frame)

            # Store reference frame for SAM (use frame 0 or update periodically)
            if strategy.frame_type == FrameType.FULL_IMAGE:
                self._current_full_image = frame.copy()
                self._current_image_shape = frame.shape[:2]

            # Process according to strategy
            self._process_frame(frame, strategy)

            # Yield results at cycle end
            if strategy.is_cycle_end:
                cycle_result = self._complete_cycle(global_frame_idx)
                yield cycle_result

    def _process_frame(self, frame: np.ndarray, strategy: FrameStrategy) -> None:
        """Process a single frame according to strategy."""
        if strategy.frame_type == FrameType.FULL_IMAGE:
            self._process_full_image(frame)
        else:
            self._process_tile(strategy)

    def _process_full_image(self, image: np.ndarray) -> None:
        """
        Process full image frame.

        Detections are in scaled coordinates if image was scaled.
        """
        # Prepare image (possibly scale down)
        processed_image = self._tile_manager.prepare_full_image(image)

        # Run detection
        detector_output = self._detector.detect(processed_image)

        # Accumulate results
        self._tile_manager.accumulate_full_image(
            anomalies=detector_output.anomalies,
            structures=detector_output.structures,
            image=processed_image,
        )

    def _process_tile(self, strategy: FrameStrategy) -> None:
        """
        Process tile frame.

        IMPORTANT: Detections are in TILE-LOCAL coordinates.
        The tile_info is preserved for SAM processing later.
        """
        tile_info = strategy.tile_info
        tile_image = tile_info.tile_image

        # Run detection on tile
        detector_output = self._detector.detect(tile_image)

        # Accumulate with tile metadata
        self._tile_manager.accumulate_tile(
            anomalies=detector_output.anomalies,
            structures=detector_output.structures,
            tile_info=tile_info,
        )

    def _complete_cycle(self, end_frame: int) -> CycleOutput:
        """
        Complete processing cycle: segment, associate, extract geometry.

        This is where SAM runs with correct coordinate handling:
        - Tile detections: SAM receives tile_image + tile-local bbox
        - Full image detections: SAM receives full image + global bbox
        """
        start_time = time.perf_counter()
        detector_time = 0.0
        segmenter_time = 0.0

        # Get merged detections from tile manager
        cycle_result = self._tile_manager.complete_cycle()

        h, w = self._current_image_shape
        global_shape = (h, w)

        # Segment all anomalies with correct coordinate handling
        seg_start = time.perf_counter()
        self._segment_accumulated_detections(cycle_result.anomalies, global_shape)
        segmenter_time = (time.perf_counter() - seg_start) * 1000

        # Build structure detections with global coordinates
        structure_detections = [
            Detection(
                detection_id=acc.detection.detection_id,
                detection_type=DetectionType.STRUCTURE,
                class_name=acc.detection.class_name,
                confidence=acc.detection.confidence,
                bbox=acc.global_bbox,
            )
            for acc in cycle_result.structures
        ]

        # Build anomaly detections for matching (global coords)
        anomaly_detections = [
            Detection(
                detection_id=acc.detection.detection_id,
                detection_type=DetectionType.ANOMALY,
                class_name=acc.detection.class_name,
                confidence=acc.detection.confidence,
                bbox=acc.global_bbox,
            )
            for acc in cycle_result.anomalies
            if acc.global_mask is not None
        ]

        # Association
        match_results = self._matcher.match(anomaly_detections, structure_detections)
        match_map = {m.anomaly_id: m for m in match_results}

        # Build final anomaly results
        anomaly_results = []
        timestamp = datetime.now()
        frame_id = f"cycle_{cycle_result.cycle_number:04d}"

        for acc in cycle_result.anomalies:
            if acc.global_mask is None:
                continue

            # Create mask with global coordinates
            mask = SegmentationMask(
                mask=acc.global_mask,
                sam_score=acc.sam_score,
            )

            # Get match info
            match_info = match_map.get(acc.detection.detection_id)

            # Extract geometry from global mask
            geometry = self._geometry.extract(mask)

            # Optionally save mask
            if self.config.save_masks and self.config.mask_output_dir:
                mask_path = self._save_mask(mask, acc.detection.detection_id, frame_id)
                mask = SegmentationMask(
                    mask=mask.mask,
                    mask_path=mask_path,
                    sam_score=mask.sam_score,
                )

            anomaly = AnomalyResult(
                anomaly_id=generate_id("anom"),
                frame_id=frame_id,
                timestamp=timestamp,
                defect_type=acc.detection.class_name,
                structure_type=match_info.structure_class if match_info else None,
                structure_id=match_info.structure_id if match_info else None,
                bbox=acc.global_bbox,
                mask=mask,
                geometry=geometry,
                detection_confidence=acc.detection.confidence,
                segmentation_confidence=acc.sam_score,
                association_confidence=match_info.confidence if match_info else None,
            )
            anomaly_results.append(anomaly)

        total_time = (time.perf_counter() - start_time) * 1000

        cycle_length = self._tile_manager.config.cycle_length
        start_frame = end_frame - cycle_length + 1

        return CycleOutput(
            cycle_id=frame_id,
            cycle_number=cycle_result.cycle_number,
            start_frame=start_frame,
            end_frame=end_frame,
            image_width=w,
            image_height=h,
            timestamp=timestamp,
            anomalies=anomaly_results,
            structures=structure_detections,
            total_time_ms=total_time,
            detector_time_ms=detector_time,
            segmenter_time_ms=segmenter_time,
        )

    def _segment_accumulated_detections(
        self,
        detections: list[AccumulatedDetection],
        global_shape: tuple[int, int],
    ) -> None:
        """
        Run SAM on accumulated detections with correct coordinate handling.

        CRITICAL:
        - For tile detections: Use tile_info.tile_image + tile-local bbox
        - For full image detections: Use full image + global bbox

        Masks are transformed to global coordinates after segmentation.
        """
        for acc in detections:
            try:
                if acc.tile_info is not None:
                    # TILE DETECTION: SAM needs tile image + tile-local bbox
                    self._segmenter.set_image(acc.tile_info.tile_image)

                    # detection.bbox is already tile-local
                    seg_result = self._segmenter.segment_detection(acc.detection)

                    if seg_result.success:
                        local_mask = seg_result.mask.mask
                        # Transform mask to global coordinates
                        global_mask = acc.tile_info.local_to_global_mask(
                            local_mask, global_shape
                        )
                        acc.local_mask = local_mask
                        acc.global_mask = global_mask
                        acc.sam_score = seg_result.mask.sam_score

                else:
                    # FULL IMAGE DETECTION: SAM uses full image
                    # May need to handle scaling if full image was scaled
                    scale = self._tile_manager.config.full_image_scale

                    if scale != 1.0:
                        # Create scaled detection for SAM
                        import cv2
                        scaled_image = cv2.resize(
                            self._current_full_image,
                            None,
                            fx=scale,
                            fy=scale,
                            interpolation=cv2.INTER_LINEAR,
                        )
                        self._segmenter.set_image(scaled_image)

                        # Scale bbox for SAM
                        scaled_det = Detection(
                            detection_id=acc.detection.detection_id,
                            detection_type=acc.detection.detection_type,
                            class_name=acc.detection.class_name,
                            confidence=acc.detection.confidence,
                            bbox=BoundingBox(
                                x_min=acc.global_bbox.x_min * scale,
                                y_min=acc.global_bbox.y_min * scale,
                                x_max=acc.global_bbox.x_max * scale,
                                y_max=acc.global_bbox.y_max * scale,
                            ),
                        )
                        seg_result = self._segmenter.segment_detection(scaled_det)

                        if seg_result.success:
                            # Unscale mask
                            scaled_mask = seg_result.mask.mask
                            global_mask = cv2.resize(
                                scaled_mask,
                                (global_shape[1], global_shape[0]),
                                interpolation=cv2.INTER_NEAREST,
                            )
                            acc.local_mask = scaled_mask
                            acc.global_mask = global_mask
                            acc.sam_score = seg_result.mask.sam_score
                    else:
                        # No scaling, use full image directly
                        self._segmenter.set_image(self._current_full_image)

                        # Create detection with global bbox
                        global_det = Detection(
                            detection_id=acc.detection.detection_id,
                            detection_type=acc.detection.detection_type,
                            class_name=acc.detection.class_name,
                            confidence=acc.detection.confidence,
                            bbox=acc.global_bbox,
                        )
                        seg_result = self._segmenter.segment_detection(global_det)

                        if seg_result.success:
                            acc.local_mask = seg_result.mask.mask
                            acc.global_mask = seg_result.mask.mask
                            acc.sam_score = seg_result.mask.sam_score

                self._segmenter.clear_image()

            except Exception as e:
                print(f"Segmentation failed for {acc.detection.detection_id}: {e}")
                continue

    def _save_mask(
        self,
        mask: SegmentationMask,
        detection_id: str,
        frame_id: str,
    ) -> Path:
        """Save mask to disk."""
        import cv2

        output_dir = self.config.mask_output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{frame_id}_{detection_id}_mask.png"
        path = output_dir / filename

        cv2.imwrite(str(path), mask.mask * 255)
        return path

    def __enter__(self):
        """Context manager entry."""
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.unload()
        return False
