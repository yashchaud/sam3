"""
Phase-1 Pipeline Orchestrator.

Coordinates the detection, segmentation, association, and geometry
extraction stages into a single coherent processing pipeline.

Supports two processing modes:
    1. Standard: For images that fit in detector input size
    2. Tiled: For large images that need to be split into tiles

Models used:
    - RF-DETR Medium: 54.7 mAP, 4.52ms latency, Apache 2.0 license
    - SAM3: Latest segmentation model (848M params), SAM license

Tiling Coordinate Handling:
    When processing tiles, coordinates flow as follows:
    1. RF-DETR outputs tile-LOCAL bounding boxes
    2. SAM receives tile image + tile-LOCAL bbox (NOT global!)
    3. SAM outputs tile-LOCAL mask
    4. Masks and bboxes are transformed to GLOBAL coordinates for output
"""

from dataclasses import dataclass
from typing import Optional
from pathlib import Path
from datetime import datetime
import numpy as np
import time

from anomaly_detection.models.data_models import (
    Detection,
    DetectionType,
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
from anomaly_detection.tiling.tiler import TileConfig, ImageTiler
from anomaly_detection.tiling.coordinator import TiledDetectionCoordinator


@dataclass
class PipelineConfig:
    """
    Configuration for the Phase-1 pipeline.

    This config aggregates settings for all pipeline components.
    For RF-DETR, you can either use COCO pretrained weights (default)
    or provide custom fine-tuned weights.

    Attributes:
        segmenter_model_path: Path to SAM3 weights file
        detector_variant: RF-DETR model size (default: MEDIUM)
        detector_weights: Optional path to custom detector weights
        detector_num_classes: Number of classes if using custom weights
        detector_config: Full detector config (overrides above if provided)
        segmenter_config: Full segmenter config (overrides model_path if provided)
        matcher_config: Structure-defect matching config
        geometry_config: Geometry extraction config
        device: Compute device ('cuda', 'cpu', or 'auto')
        save_masks: Whether to save masks to disk
        mask_output_dir: Directory for mask outputs (required if save_masks=True)

    Example:
        # Simple config with defaults
        config = PipelineConfig(
            segmenter_model_path=Path("weights/sam3.pt"),
        )

        # Config with custom detector weights
        config = PipelineConfig(
            segmenter_model_path=Path("weights/sam3.pt"),
            detector_weights=Path("weights/rfdetr_structural.pth"),
            detector_num_classes=22,
        )
    """
    # Required: SAM3 model path
    segmenter_model_path: Path

    # Detector settings (RF-DETR)
    detector_variant: RFDETRVariant = RFDETRVariant.MEDIUM
    detector_weights: Optional[Path] = None  # None = use COCO pretrained
    detector_num_classes: Optional[int] = None

    # Full component configs (optional - override individual settings)
    detector_config: Optional[DetectorConfig] = None
    segmenter_config: Optional[SegmenterConfig] = None
    matcher_config: Optional[MatcherConfig] = None
    geometry_config: Optional[GeometryConfig] = None

    # Pipeline settings
    device: str = "auto"
    save_masks: bool = False
    mask_output_dir: Optional[Path] = None

    # Tiling settings for large images
    enable_tiling: bool = True  # Auto-tile large images
    tile_config: Optional[TileConfig] = None
    nms_iou_threshold: float = 0.5  # For deduplicating overlapping detections

    def __post_init__(self):
        # Build detector config if not provided
        if self.detector_config is None:
            self.detector_config = DetectorConfig(
                variant=self.detector_variant,
                pretrain_weights=self.detector_weights,
                num_classes=self.detector_num_classes,
                device=self.device,
            )

        # Build segmenter config if not provided
        if self.segmenter_config is None:
            self.segmenter_config = SegmenterConfig(
                model_path=self.segmenter_model_path,
                device=self.device,
            )

        # Use default configs for matcher and geometry
        if self.matcher_config is None:
            self.matcher_config = MatcherConfig()

        if self.geometry_config is None:
            self.geometry_config = GeometryConfig()

        # Validate mask saving config
        if self.save_masks and self.mask_output_dir is None:
            raise ValueError("mask_output_dir required when save_masks=True")

        # Default tile config
        if self.tile_config is None:
            self.tile_config = TileConfig()


class Phase1Pipeline:
    """
    Main orchestrator for the Phase-1 anomaly detection pipeline.

    Processes images through four stages:
        1. Detection (RF-DETR) - finds structures and anomaly candidates
        2. Segmentation (SAM3) - generates pixel-accurate masks for anomalies
        3. Association - links anomalies to parent structural elements
        4. Geometry - extracts dimensional properties from masks

    Models:
        - RF-DETR Medium: State-of-the-art detection (54.7 mAP), Apache 2.0
        - SAM3: Latest segmentation model from Meta, SAM license

    Usage:
        config = PipelineConfig(
            segmenter_model_path=Path("weights/sam3.pt"),
            detector_variant=RFDETRVariant.MEDIUM,
        )

        # Using context manager (recommended)
        with Phase1Pipeline(config) as pipeline:
            result = pipeline.process(image)
            for anomaly in result.anomalies:
                print(f"{anomaly.defect_type} on {anomaly.structure_type}")

        # Or manual load/unload
        pipeline = Phase1Pipeline(config)
        pipeline.load()
        try:
            result = pipeline.process(image)
        finally:
            pipeline.unload()
    """

    def __init__(self, config: PipelineConfig):
        """
        Initialize the pipeline.

        Args:
            config: Pipeline configuration

        Note:
            This only initializes components. Call load() to load models.
        """
        self.config = config

        # Initialize components (not loaded yet)
        self._detector = RFDETRDetector(config.detector_config)
        self._segmenter = SAM3Segmenter(config.segmenter_config)
        self._matcher = StructureDefectMatcher(config.matcher_config)
        self._geometry = MaskGeometryExtractor(config.geometry_config)

        # Tiling support
        self._tiler = ImageTiler(config.tile_config)
        self._tiled_coordinator: Optional[TiledDetectionCoordinator] = None

        self._loaded = False

    def load(self) -> None:
        """
        Load all models into memory.

        Call this before processing. Separating load from __init__
        gives explicit control over when heavy model loading occurs.

        Raises:
            RuntimeError: If model loading fails
        """
        self._detector.load()
        self._segmenter.load()

        # Initialize tiled coordinator (needs loaded models)
        if self.config.enable_tiling:
            self._tiled_coordinator = TiledDetectionCoordinator(
                detector=self._detector,
                segmenter=self._segmenter,
                tile_config=self.config.tile_config,
                nms_iou_threshold=self.config.nms_iou_threshold,
            )

        self._loaded = True

    def is_loaded(self) -> bool:
        """Check if pipeline is ready for processing."""
        return self._loaded

    def process(
        self,
        image: np.ndarray,
        frame_id: Optional[str] = None,
        source_path: Optional[Path] = None,
        timestamp: Optional[datetime] = None,
    ) -> PipelineOutput:
        """
        Process a single image through the full pipeline.

        Args:
            image: Input image as numpy array (H, W, C) in RGB format
            frame_id: Optional identifier for this frame
            source_path: Optional source file path for metadata
            timestamp: Optional timestamp (defaults to now)

        Returns:
            PipelineOutput containing:
                - anomalies: List of detected and processed anomalies
                - structures: List of detected structural elements
                - processing times and metadata

        Raises:
            RuntimeError: If pipeline not loaded
            ValueError: If image format is invalid
        """
        if not self._loaded:
            raise RuntimeError("Pipeline not loaded. Call load() first.")

        start_time = time.perf_counter()

        # Generate IDs and timestamp
        frame_id = frame_id or generate_id("frame")
        timestamp = timestamp or datetime.now()

        # Check if tiling is needed
        use_tiling = (
            self.config.enable_tiling
            and self._tiled_coordinator is not None
            and self._tiler.should_tile(image)
        )

        if use_tiling:
            return self._process_tiled(
                image=image,
                frame_id=frame_id,
                source_path=source_path,
                timestamp=timestamp,
                start_time=start_time,
            )
        else:
            return self._process_standard(
                image=image,
                frame_id=frame_id,
                source_path=source_path,
                timestamp=timestamp,
                start_time=start_time,
            )

    def _process_standard(
        self,
        image: np.ndarray,
        frame_id: str,
        source_path: Optional[Path],
        timestamp: datetime,
        start_time: float,
    ) -> PipelineOutput:
        """
        Standard processing for images that fit in detector input size.

        This is the original processing path - no tiling involved.
        """
        # Stage 1: Detection
        detector_output = self._detector.detect(image)
        detector_time = detector_output.inference_time_ms

        # Stage 2: Segmentation (only for anomalies)
        segmenter_output = self._segmenter.segment_batch(
            image,
            detector_output.anomalies,
        )
        segmenter_time = segmenter_output.inference_time_ms

        # Build detection -> mask mapping
        mask_map = {
            result.detection_id: result.mask
            for result in segmenter_output.results
            if result.success
        }

        # Stage 3: Association
        match_results = self._matcher.match(
            detector_output.anomalies,
            detector_output.structures,
        )
        match_map = {m.anomaly_id: m for m in match_results}

        # Stage 4: Geometry extraction and result assembly
        anomaly_results = []

        for detection in detector_output.anomalies:
            # Get mask
            mask = mask_map.get(detection.detection_id)
            if mask is None:
                # Skip anomalies without valid masks
                continue

            # Get match info
            match_info = match_map.get(detection.detection_id)

            # Extract geometry
            geometry = self._geometry.extract(mask)

            # Optionally save mask
            if self.config.save_masks:
                mask_path = self._save_mask(mask, detection.detection_id, frame_id)
                mask = SegmentationMask(
                    mask=mask.mask,
                    mask_path=mask_path,
                    sam_score=mask.sam_score,
                )

            # Build result
            anomaly = AnomalyResult(
                anomaly_id=generate_id("anom"),
                frame_id=frame_id,
                timestamp=timestamp,
                defect_type=detection.class_name,
                structure_type=match_info.structure_class if match_info else None,
                structure_id=match_info.structure_id if match_info else None,
                bbox=detection.bbox,
                mask=mask,
                geometry=geometry,
                detection_confidence=detection.confidence,
                segmentation_confidence=mask.sam_score,
                association_confidence=match_info.confidence if match_info else None,
            )
            anomaly_results.append(anomaly)

        total_time = (time.perf_counter() - start_time) * 1000

        return PipelineOutput(
            frame_id=frame_id,
            source_path=source_path,
            image_width=image.shape[1],
            image_height=image.shape[0],
            timestamp=timestamp,
            anomalies=anomaly_results,
            structures=detector_output.structures,
            processing_time_ms=total_time,
            detector_time_ms=detector_time,
            segmenter_time_ms=segmenter_time,
        )

    def _process_tiled(
        self,
        image: np.ndarray,
        frame_id: str,
        source_path: Optional[Path],
        timestamp: datetime,
        start_time: float,
    ) -> PipelineOutput:
        """
        Tiled processing for large images.

        Coordinate handling:
        1. Image is split into overlapping tiles
        2. RF-DETR runs on each tile -> tile-local bboxes
        3. SAM receives TILE IMAGE + TILE-LOCAL bbox (critical!)
        4. SAM outputs tile-local mask
        5. Masks/bboxes are transformed to global coordinates
        6. NMS removes duplicates from tile overlaps
        """
        global_h, global_w = image.shape[:2]
        global_shape = (global_h, global_w)

        # Run tiled detection + segmentation
        # The coordinator handles all coordinate transforms correctly
        seg_results, structure_detections = self._tiled_coordinator.process_anomalies_only(image)

        detector_time = 0.0  # Accumulated across tiles
        segmenter_time = 0.0

        # Stage 3: Association using GLOBAL coordinates
        # Convert structure TiledDetections to regular Detections for matcher
        # (structures already have global_bbox from coordinator)

        # Build anomaly detections with global coords for matching
        anomaly_detections_global = []
        for seg_result in seg_results:
            det = Detection(
                detection_id=seg_result.detection.detection_id,
                detection_type=DetectionType.ANOMALY,
                class_name=seg_result.detection.class_name,
                confidence=seg_result.detection.confidence,
                bbox=seg_result.global_bbox,  # Use GLOBAL bbox for matching
            )
            anomaly_detections_global.append(det)

        match_results = self._matcher.match(
            anomaly_detections_global,
            structure_detections,  # Already have global coords
        )
        match_map = {m.anomaly_id: m for m in match_results}

        # Stage 4: Geometry extraction and result assembly
        anomaly_results = []

        for seg_result in seg_results:
            detection = seg_result.detection

            # Create mask with GLOBAL coordinates
            mask = SegmentationMask(
                mask=seg_result.global_mask,  # Use GLOBAL mask
                sam_score=seg_result.sam_score,
            )

            # Get match info
            match_info = match_map.get(detection.detection_id)

            # Extract geometry from GLOBAL mask
            geometry = self._geometry.extract(mask)

            # Optionally save mask
            if self.config.save_masks:
                mask_path = self._save_mask(mask, detection.detection_id, frame_id)
                mask = SegmentationMask(
                    mask=mask.mask,
                    mask_path=mask_path,
                    sam_score=mask.sam_score,
                )

            # Build result with GLOBAL bbox
            anomaly = AnomalyResult(
                anomaly_id=generate_id("anom"),
                frame_id=frame_id,
                timestamp=timestamp,
                defect_type=detection.class_name,
                structure_type=match_info.structure_class if match_info else None,
                structure_id=match_info.structure_id if match_info else None,
                bbox=seg_result.global_bbox,  # GLOBAL coordinates
                mask=mask,
                geometry=geometry,
                detection_confidence=detection.confidence,
                segmentation_confidence=seg_result.sam_score,
                association_confidence=match_info.confidence if match_info else None,
            )
            anomaly_results.append(anomaly)

        total_time = (time.perf_counter() - start_time) * 1000

        return PipelineOutput(
            frame_id=frame_id,
            source_path=source_path,
            image_width=global_w,
            image_height=global_h,
            timestamp=timestamp,
            anomalies=anomaly_results,
            structures=structure_detections,
            processing_time_ms=total_time,
            detector_time_ms=detector_time,
            segmenter_time_ms=segmenter_time,
        )

    def process_batch(
        self,
        images: list[np.ndarray],
        frame_ids: Optional[list[str]] = None,
    ) -> list[PipelineOutput]:
        """
        Process multiple images sequentially.

        Note: This is a simple sequential implementation.
        Batched GPU inference can be added in later phases.

        Args:
            images: List of input images (H, W, C) in RGB format
            frame_ids: Optional list of frame identifiers

        Returns:
            List of PipelineOutput, one per image

        Raises:
            ValueError: If frame_ids length doesn't match images length
        """
        if frame_ids is None:
            frame_ids = [None] * len(images)

        if len(frame_ids) != len(images):
            raise ValueError(
                f"frame_ids length ({len(frame_ids)}) != images length ({len(images)})"
            )

        return [
            self.process(image, frame_id=fid)
            for image, fid in zip(images, frame_ids)
        ]

    def _save_mask(
        self,
        mask: SegmentationMask,
        detection_id: str,
        frame_id: str,
    ) -> Path:
        """Save mask to disk and return path."""
        import cv2

        output_dir = self.config.mask_output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{frame_id}_{detection_id}_mask.png"
        path = output_dir / filename

        cv2.imwrite(str(path), mask.mask * 255)
        return path

    def unload(self) -> None:
        """
        Release all models from memory.

        Frees GPU memory and clears internal state.
        """
        self._detector.unload()
        self._segmenter.unload()
        self._loaded = False

    def __enter__(self):
        """Context manager entry - loads models."""
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - unloads models."""
        self.unload()
        return False

    def __repr__(self) -> str:
        """Return string representation of pipeline state."""
        status = "loaded" if self._loaded else "not loaded"
        return f"Phase1Pipeline(status={status}, detector={self._detector})"
