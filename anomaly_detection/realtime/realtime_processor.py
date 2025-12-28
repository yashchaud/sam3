"""Real-time video processor with SAM3-first pipeline + VLM judging."""

import asyncio
import time
import logging
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable
from collections import deque
import threading

from .config import RealtimeConfig, ProcessingStats, FrameSource
from .frame_buffer import FrameBuffer, BufferedFrame
from .stream_handler import StreamHandler, StreamState

from ..vlm import VLMJudge, VLMConfig, VLMResponse, VLMPrediction, PredictionType
from ..segmenter import SAM3Segmenter, SegmenterConfig
from ..models import Detection, AnomalyResult, PipelineOutput, BoundingBox, SegmentationMask
from ..geometry import MaskGeometryExtractor
from ..config import DEFAULT_ANOMALY_CLASSES
from ..debug_output import DebugOutputManager

# Setup logger for VLM predictions
logger = logging.getLogger(__name__)


@dataclass
class SAMCandidate:
    """Candidate segmentation from SAM3."""
    defect_type: str
    mask: np.ndarray
    bbox: BoundingBox
    confidence: float = 0.5


@dataclass
class FrameResult:
    """Result from processing a single frame."""
    frame_id: str
    frame_index: int
    timestamp: float

    # Detection results
    anomalies: list[AnomalyResult] = field(default_factory=list)
    structures: list[Detection] = field(default_factory=list)

    # VLM-judged anomalies (SAM3 candidates that passed VLM filtering)
    vlm_judged_anomalies: list[AnomalyResult] = field(default_factory=list)
    vlm_response: VLMResponse | None = None

    # SAM3 candidate count
    sam_candidate_count: int = 0

    # Timing
    detection_time_ms: float = 0.0
    segmentation_time_ms: float = 0.0
    vlm_time_ms: float = 0.0
    total_time_ms: float = 0.0

    # VLM pending - if True, VLM is still processing and will update later
    vlm_pending: bool = False

    # Pending VLM task for later retrieval
    _vlm_task: asyncio.Task | None = field(default=None, repr=False)

    @property
    def all_anomalies(self) -> list[AnomalyResult]:
        return self.anomalies + self.vlm_judged_anomalies

    def to_dict(self) -> dict:
        return {
            "frame_id": self.frame_id,
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
            "anomaly_count": len(self.all_anomalies),
            "structure_count": len(self.structures),
            "sam_candidates": self.sam_candidate_count,
            "vlm_judged_count": len(self.vlm_judged_anomalies),
            "vlm_pending": self.vlm_pending,
            "timing": {
                "detection_ms": round(self.detection_time_ms, 2),
                "segmentation_ms": round(self.segmentation_time_ms, 2),
                "vlm_ms": round(self.vlm_time_ms, 2),
                "total_ms": round(self.total_time_ms, 2),
            },
        }


class RealtimeVideoProcessor:
    """
    Real-time video processor with SAM3-first pipeline + VLM judging.

    Pipeline:
    1. SAM3 segments ALL default anomaly classes (Crack, Corrosion, etc.) as text prompts
    2. Submit SAM3 candidates to VLM for judging (async)
    3. VLM judges which candidates are real anomalies
    4. Only keep VLM-approved detections

    Supports:
    - Video files (MP4, AVI, etc.)
    - Webcam/live camera streams
    - RTSP streams
    - Image sequences
    """

    def __init__(self, config: RealtimeConfig, debug_output_dir: str | Path | None = None):
        self.config = config

        # Initialize components (SAM3 + VLM Judge only, no RF-DETR)
        self._segmenter: SAM3Segmenter | None = None
        self._vlm_judge: VLMJudge | None = None
        self._video_tracker = None  # SAM3VideoTracker for efficient video processing
        self._geometry = MaskGeometryExtractor()

        # Stream handling
        self._stream: StreamHandler | None = None
        self._buffer: FrameBuffer | None = None

        # State
        self._is_loaded = False
        self._is_running = False
        self._current_frame_index = 0

        # Statistics
        self._stats = ProcessingStats()
        self._processing_times: deque[float] = deque(maxlen=100)

        # Pending SAM candidates awaiting VLM judgment (frame_id -> candidates)
        self._pending_sam_candidates: dict[str, list[SAMCandidate]] = {}

        # Result callbacks
        self._on_result_callback: Callable[[FrameResult], None] | None = None

        # Debug output
        self._debug = DebugOutputManager(
            output_dir=debug_output_dir or "output/debug",
            enabled=debug_output_dir is not None
        )

    def load(self) -> None:
        """Load all models into memory."""
        if self._is_loaded:
            return

        # Initialize segmenter (SAM3)
        segmenter_config = self.config.get_segmenter_config()
        self._segmenter = SAM3Segmenter(segmenter_config)
        self._segmenter.load()
        logger.info("SAM3 segmenter loaded")

        # Initialize video tracker (SAM3TrackerModel for efficient video processing)
        try:
            from ..segmenter.sam3_video_tracker import SAM3VideoTracker
            self._video_tracker = SAM3VideoTracker(segmenter_config)
            self._video_tracker.load()
            logger.info("SAM3 Video Tracker loaded")
        except ImportError as e:
            logger.warning(f"SAM3 Video Tracker not available: {e}. Will use frame-by-frame processing.")
            self._video_tracker = None

        # Initialize VLM judge (required for this pipeline)
        vlm_config = self.config.get_vlm_config()
        self._vlm_judge = VLMJudge(vlm_config)
        self._vlm_judge.load()
        logger.info(f"VLM Judge loaded (provider: {vlm_config.provider.value}, model: {vlm_config.openrouter_model})")

        self._is_loaded = True

    def unload(self) -> None:
        """Unload all models from memory."""
        if self._vlm_judge:
            self._vlm_judge.unload()
            self._vlm_judge = None

        if self._segmenter:
            self._segmenter.unload()
            self._segmenter = None

        self._is_loaded = False

    def is_loaded(self) -> bool:
        return self._is_loaded

    async def process_video(
        self,
        source_path: str | None = None,
        source_type: FrameSource | None = None,
    ) -> AsyncIterator[FrameResult]:
        """
        Process a video and yield results for each frame.

        Args:
            source_path: Override source path from config
            source_type: Override source type from config

        Yields:
            FrameResult for each processed frame
        """
        if not self._is_loaded:
            raise RuntimeError("Models not loaded. Call load() first.")

        # Setup stream
        self._buffer = FrameBuffer(max_size=self.config.frame_buffer_size)
        self._stream = StreamHandler(
            source_type=source_type or self.config.source_type,
            source_path=source_path or self.config.source_path,
            webcam_id=self.config.webcam_id,
            target_fps=self.config.target_fps,
            buffer=self._buffer,
        )

        if not self._stream.open():
            raise RuntimeError(f"Failed to open video source: {source_path}")

        self._is_running = True
        self._current_frame_index = 0

        try:
            # Process frames
            for buffered_frame in self._stream.iter_frames():
                if not self._is_running:
                    break

                result = await self._process_frame(buffered_frame)

                # Update stats
                self._stats.total_frames_processed += 1
                self._stats.total_detections += len(result.all_anomalies)
                self._processing_times.append(result.total_time_ms)

                if self._on_result_callback:
                    self._on_result_callback(result)

                yield result

                self._current_frame_index += 1

        finally:
            self._is_running = False
            if self._stream:
                self._stream.close()

    async def process_single_image(
        self,
        image: np.ndarray,
        frame_id: str | None = None,
        on_update: Callable[[FrameResult, np.ndarray], None] | None = None,
    ) -> FrameResult:
        """
        Process a single image with sequential tile processing and incremental updates.

        For single images, we process tiles sequentially and call on_update after each
        step so the UI can show progress incrementally.

        Args:
            image: RGB image (HWC)
            frame_id: Optional frame identifier
            on_update: Callback(result, image) called after each processing step

        Returns:
            FrameResult with all detections
        """
        if not self._is_loaded:
            raise RuntimeError("Models not loaded. Call load() first.")

        frame_id = frame_id or f"image_{self._current_frame_index:08d}"

        buffered = BufferedFrame(
            image=image,
            frame_index=self._current_frame_index,
            timestamp=time.time(),
            frame_id=frame_id,
        )

        # Process with sequential tile updates
        result = await self._process_single_image_sync(buffered, on_update=on_update)
        self._current_frame_index += 1

        return result

    async def _process_single_image_sync(
        self,
        frame: BufferedFrame,
        on_update: Callable[[FrameResult, np.ndarray], None] | None = None,
    ) -> FrameResult:
        """
        Process a single image with SEQUENTIAL tile processing and incremental updates.

        Flow:
        1. SAM3 Global scan -> update UI
        2. For each tile: SAM3 scan -> accumulate masks -> update UI
        3. VLM runs in background, updates when ready

        Each step sends an update via on_update callback so UI shows progress.

        Args:
            frame: The frame to process
            on_update: Callback(result, image) called after each processing step
        """
        start_time = time.perf_counter()

        # Initialize result
        result = FrameResult(
            frame_id=frame.frame_id,
            frame_index=frame.frame_index,
            timestamp=frame.timestamp,
        )

        # Track all accumulated masks
        all_masks = []

        # Debug: Save original frame
        self._debug.save_original_frame(frame.frame_id, frame.frame_index, frame.image)

        # Get defect classes for SAM3 text prompts
        anomaly_classes = [cls.title() for cls in DEFAULT_ANOMALY_CLASSES]
        logger.info(f"[Frame {frame.frame_index}] Starting SEQUENTIAL pipeline with {len(anomaly_classes)} defect classes")

        # Generate grid overlay for VLM (but SAM3 uses original image!)
        self._vlm_judge.grid.compute_grid(frame.image.shape[1], frame.image.shape[0])
        grid_image = self._vlm_judge.grid.draw_grid(frame.image)
        self._debug.save_vlm_grid(frame.frame_id, frame.frame_index, grid_image)

        grid_config = self._vlm_judge.grid.config

        # Start VLM in background (non-blocking network I/O)
        vlm_task = asyncio.create_task(
            self._vlm_judge.process_frame(
                frame.image,
                frame.frame_id,
                frame.frame_index,
            )
        )
        result.vlm_pending = True
        result._vlm_task = vlm_task

        # ========================================
        # STEP 1: Global SAM3 scan
        # ========================================
        sam_global_start = time.perf_counter()
        logger.info(f"[Frame {frame.frame_index}] SAM3 Global: Running...")

        global_results = self._segmenter.segment_with_text_batch(
            frame.image,
            anomaly_classes,
            prefix=f"{frame.frame_id}_global"
        )
        sam_global_time = (time.perf_counter() - sam_global_start) * 1000

        # Add global results to accumulated masks
        for res in global_results:
            if res.success and res.mask:
                defect_type = res.detection_id.split('_')[-2] if '_' in res.detection_id else "unknown"
                all_masks.append({
                    'mask': res.mask.data,
                    'score': res.mask.sam_score,
                    'defect_type': defect_type,
                    'source': 'global',
                    'vlm_confidence': None,
                })

        logger.info(f"[Frame {frame.frame_index}] SAM3 Global: {len(global_results)} masks in {sam_global_time:.0f}ms")

        # Update result and send to UI
        self._update_result_from_masks(result, all_masks, frame)
        result.segmentation_time_ms = sam_global_time
        result.total_time_ms = (time.perf_counter() - start_time) * 1000

        if on_update:
            await self._call_update_async(on_update, result, frame.image)

        # ========================================
        # STEP 2: Per-tile SAM3 scans (sequential)
        # ========================================
        h, w = frame.image.shape[:2]
        tile_h = h // grid_config.rows
        tile_w = w // grid_config.cols

        sam_tile_time = 0.0

        for row in range(grid_config.rows):
            for col in range(grid_config.cols):
                tile_label = f"{chr(65 + row)}{col + 1}"
                tile_start = time.perf_counter()

                # Extract tile
                y_start = row * tile_h
                x_start = col * tile_w
                y_end = h if row == grid_config.rows - 1 else (row + 1) * tile_h
                x_end = w if col == grid_config.cols - 1 else (col + 1) * tile_w

                tile = frame.image[y_start:y_end, x_start:x_end]

                # Process tile with all prompts
                tile_results = self._segmenter.segment_with_text_batch(
                    tile,
                    anomaly_classes,
                    prefix=f"{frame.frame_id}_tile_{tile_label}"
                )

                tile_time = (time.perf_counter() - tile_start) * 1000
                sam_tile_time += tile_time

                # Add tile results to accumulated masks (transform to full image coords)
                new_masks_count = 0
                for res in tile_results:
                    if res.success and res.mask:
                        defect_type = res.detection_id.split('_')[-2] if '_' in res.detection_id else "unknown"

                        # Transform mask to full image coordinates
                        full_mask = np.zeros((h, w), dtype=np.uint8)
                        mask_h, mask_w = res.mask.data.shape[:2]
                        full_mask[y_start:y_start + mask_h, x_start:x_start + mask_w] = res.mask.data

                        all_masks.append({
                            'mask': full_mask,
                            'score': res.mask.sam_score,
                            'defect_type': defect_type,
                            'source': f'tile_{tile_label}',
                            'vlm_confidence': None,
                        })
                        new_masks_count += 1

                logger.info(f"[Frame {frame.frame_index}] SAM3 Tile {tile_label}: {new_masks_count} masks in {tile_time:.0f}ms")

                # Update result with all accumulated masks and send to UI
                self._update_result_from_masks(result, all_masks, frame)
                result.segmentation_time_ms = sam_global_time + sam_tile_time
                result.total_time_ms = (time.perf_counter() - start_time) * 1000

                if on_update:
                    await self._call_update_async(on_update, result, frame.image)

                # Brief yield to allow other async tasks (like VLM) to progress
                await asyncio.sleep(0)

        logger.info(f"[Frame {frame.frame_index}] SAM3 Tiles complete: {sam_tile_time:.0f}ms total")

        # ========================================
        # STEP 3: Check if VLM is done
        # ========================================
        if vlm_task.done():
            try:
                vlm_response = vlm_task.result()
                result.vlm_pending = False
                result._vlm_task = None
                result.vlm_time_ms = vlm_response.generation_time_ms
                result.vlm_response = vlm_response

                logger.info(f"[Frame {frame.frame_index}] VLM complete: {len(vlm_response.predictions) if vlm_response else 0} predictions")

                if vlm_response and vlm_response.predictions:
                    # Process VLM predictions with point prompts
                    for i, pred in enumerate(vlm_response.predictions):
                        if pred.confidence < self.config.confidence_threshold:
                            continue

                        point_xy = None
                        if pred.grid_cell:
                            col, row = pred.grid_cell
                            cell = self._vlm_judge.grid.get_cell(col, row)
                            if cell:
                                point_xy = (cell.center_x, cell.center_y)

                        if point_xy:
                            seg_result = self._segmenter.segment_with_point(
                                frame.image,
                                point_xy,
                                detection_id=f"{frame.frame_id}_vlm_{pred.defect_type}_{i}"
                            )
                            if seg_result.success and seg_result.mask:
                                all_masks.append({
                                    'mask': seg_result.mask.data,
                                    'score': seg_result.mask.sam_score,
                                    'defect_type': pred.defect_type,
                                    'source': 'vlm_point',
                                    'vlm_confidence': pred.confidence,
                                })

                    # Final update with VLM results
                    self._update_result_from_masks(result, all_masks, frame)
                    if on_update:
                        await self._call_update_async(on_update, result, frame.image)

            except Exception as e:
                logger.error(f"[Frame {frame.frame_index}] VLM task failed: {e}")
                result.vlm_pending = False

        # Final timing
        result.total_time_ms = (time.perf_counter() - start_time) * 1000
        result.sam_candidate_count = len(all_masks)

        logger.info(
            f"[Frame {frame.frame_index}] Complete: {len(result.all_anomalies)} anomalies "
            f"in {result.total_time_ms:.0f}ms (VLM pending: {result.vlm_pending})"
        )

        return result

    def _update_result_from_masks(
        self,
        result: FrameResult,
        all_masks: list[dict],
        frame: BufferedFrame,
    ) -> None:
        """Update FrameResult anomalies from accumulated masks with deduplication."""
        # Deduplicate by IoU
        deduplicated = self._deduplicate_masks(all_masks, iou_threshold=0.5)

        # Clear and rebuild anomalies
        result.vlm_judged_anomalies.clear()

        for i, mask_info in enumerate(deduplicated):
            mask_data = mask_info['mask']
            bbox = self._extract_bbox_from_mask(mask_data)

            if bbox:
                geometry = self._geometry.extract(mask_data)
                anomaly = AnomalyResult(
                    anomaly_id=f"{frame.frame_id}_{mask_info['defect_type']}_{i}",
                    frame_id=frame.frame_id,
                    timestamp=time.time(),
                    defect_type=mask_info['defect_type'],
                    structure_type=None,
                    bbox=bbox,
                    mask=SegmentationMask(data=mask_data, sam_score=mask_info['score']),
                    geometry=geometry,
                    detection_confidence=mask_info['vlm_confidence'] or mask_info['score'],
                    segmentation_confidence=mask_info['score'],
                    association_confidence=1.0,
                )
                result.vlm_judged_anomalies.append(anomaly)

        result.sam_candidate_count = len(all_masks)

    async def _call_update_async(
        self,
        callback: Callable[[FrameResult, np.ndarray], None],
        result: FrameResult,
        image: np.ndarray,
    ) -> None:
        """Call update callback, handling both sync and async callbacks."""
        try:
            ret = callback(result, image)
            if asyncio.iscoroutine(ret):
                await ret
        except Exception as e:
            logger.error(f"Update callback error: {e}")

    def _deduplicate_masks(
        self,
        masks: list[dict],
        iou_threshold: float = 0.5,
    ) -> list[dict]:
        """
        Deduplicate masks using IoU (Intersection over Union).

        Keeps masks with highest score when IoU > threshold.
        VLM-sourced masks get priority (have vlm_confidence set).

        Args:
            masks: List of mask dicts with 'mask', 'score', 'defect_type', 'source', 'vlm_confidence'
            iou_threshold: Merge masks with IoU above this threshold

        Returns:
            Deduplicated list of mask dicts
        """
        if not masks:
            return []

        # Sort by priority: VLM-confirmed first, then by score
        def priority_key(m):
            vlm_boost = 1.0 if m['vlm_confidence'] else 0.0
            return (vlm_boost, m['score'])

        sorted_masks = sorted(masks, key=priority_key, reverse=True)

        kept = []
        for mask_info in sorted_masks:
            mask = mask_info['mask']

            # Check IoU with all kept masks
            is_duplicate = False
            for kept_info in kept:
                iou = self._compute_mask_iou(mask, kept_info['mask'])
                if iou > iou_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                kept.append(mask_info)

        return kept

    def _compute_mask_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Compute IoU between two binary masks."""
        if mask1.shape != mask2.shape:
            return 0.0

        intersection = np.logical_and(mask1 > 0, mask2 > 0).sum()
        union = np.logical_or(mask1 > 0, mask2 > 0).sum()

        if union == 0:
            return 0.0

        return float(intersection) / float(union)

    async def await_vlm_update(
        self,
        result: FrameResult,
        image: np.ndarray,
    ) -> FrameResult | None:
        """
        Await pending VLM task and return updated result with VLM predictions.

        Call this after process_single_image returns with vlm_pending=True.
        Returns None if VLM was not pending or failed.

        Args:
            result: The FrameResult with vlm_pending=True
            image: Original image for SAM3 point prompts

        Returns:
            Updated FrameResult with VLM predictions, or None if not applicable
        """
        if not result.vlm_pending or result._vlm_task is None:
            return None

        try:
            vlm_start = time.perf_counter()
            vlm_response = await result._vlm_task
            vlm_time = (time.perf_counter() - vlm_start) * 1000

            if not vlm_response or not vlm_response.predictions:
                logger.info(f"[Frame {result.frame_index}] VLM returned no predictions")
                result.vlm_pending = False
                result.vlm_time_ms = vlm_response.generation_time_ms if vlm_response else 0
                return result

            logger.info(f"[Frame {result.frame_index}] VLM completed: {len(vlm_response.predictions)} predictions in {vlm_time:.0f}ms")

            result.vlm_response = vlm_response
            result.vlm_time_ms = vlm_response.generation_time_ms

            # Process VLM predictions with SAM3 point prompts
            vlm_point_results = []
            for i, pred in enumerate(vlm_response.predictions):
                if pred.confidence < self.config.confidence_threshold:
                    continue

                point_xy = None
                if pred.grid_cell:
                    col, row = pred.grid_cell
                    cell = self._vlm_judge.grid.get_cell(col, row)
                    if cell:
                        point_xy = (cell.center_x, cell.center_y)

                if point_xy:
                    seg_result = self._segmenter.segment_with_point(
                        image,
                        point_xy,
                        detection_id=f"{result.frame_id}_vlm_{pred.defect_type}_{i}"
                    )
                    if seg_result.success and seg_result.mask:
                        seg_result.mask.vlm_confidence = pred.confidence
                        seg_result.mask.defect_type = pred.defect_type
                        vlm_point_results.append(seg_result)

            logger.info(f"[Frame {result.frame_index}] VLM update: {len(vlm_point_results)} new masks from VLM predictions")

            # Add VLM-prompted results to anomalies
            for res in vlm_point_results:
                if res.success and res.mask:
                    defect_type = getattr(res.mask, 'defect_type', 'unknown')
                    vlm_conf = getattr(res.mask, 'vlm_confidence', None)
                    bbox = self._extract_bbox_from_mask(res.mask.data)

                    if bbox:
                        anomaly = AnomalyResult(
                            anomaly_id=f"{result.frame_id}_vlm_{defect_type}_{len(result.vlm_judged_anomalies)}",
                            frame_id=result.frame_id,
                            timestamp=time.time(),
                            defect_type=defect_type,
                            structure_type=None,
                            bbox=bbox,
                            mask=SegmentationMask(data=res.mask.data, sam_score=res.mask.sam_score),
                            geometry=self._geometry.extract(res.mask.data),
                            detection_confidence=vlm_conf or res.mask.sam_score,
                            segmentation_confidence=res.mask.sam_score,
                            association_confidence=1.0,
                        )
                        result.vlm_judged_anomalies.append(anomaly)

            result.vlm_pending = False
            result._vlm_task = None

            return result

        except Exception as e:
            logger.error(f"[Frame {result.frame_index}] VLM update failed: {e}")
            result.vlm_pending = False
            return None

    async def _process_frame(self, frame: BufferedFrame) -> FrameResult:
        """
        Process a single buffered frame using SAM3 Video Tracker for efficient temporal tracking.

        Uses Sam3TrackerModel to track objects across frames with temporal consistency:
        - First frame: Initialize tracks with text prompts
        - Subsequent frames: Propagate existing tracks efficiently (reuses embeddings)

        Falls back to frame-by-frame processing if video tracker is unavailable.
        """
        start_time = time.perf_counter()

        # Initialize result
        result = FrameResult(
            frame_id=frame.frame_id,
            frame_index=frame.frame_index,
            timestamp=frame.timestamp,
        )

        # Debug: Save original frame
        self._debug.save_original_frame(frame.frame_id, frame.frame_index, frame.image)

        # Use video tracker if available
        if self._video_tracker and self._video_tracker.is_loaded():
            result = await self._process_frame_with_tracker(frame, result, start_time)
        else:
            # Fallback to frame-by-frame processing
            result = await self._process_frame_fallback(frame, result, start_time)

        # Debug: Save frame timing
        self._debug.save_frame_timing(
            frame.frame_id,
            frame.frame_index,
            result.segmentation_time_ms,
            result.vlm_time_ms,
            result.total_time_ms,
            result.sam_candidate_count,
            len(result.vlm_judged_anomalies)
        )

        return result

    async def _process_frame_with_tracker(
        self,
        frame: BufferedFrame,
        result: FrameResult,
        start_time: float
    ) -> FrameResult:
        """Process frame using SAM3 Video Tracker for efficient temporal tracking."""
        seg_start = time.perf_counter()

        all_anomaly_classes = [cls.title() for cls in DEFAULT_ANOMALY_CLASSES]

        # First frame: Initialize tracking with text prompts
        if frame.frame_index == 0:
            # Use all classes on first frame to establish initial tracks
            logger.info(f"[Frame 0] Initializing video tracker with {len(all_anomaly_classes)} classes")

            self._video_tracker.reset_tracking()
            tracks = self._video_tracker.initialize_with_text_prompts(
                frame.image,
                all_anomaly_classes,
                frame_idx=frame.frame_index
            )

            # Convert tracks to anomalies
            for track in tracks:
                if len(track.masks) > 0:
                    mask_data = track.masks[-1]
                    bbox = self._extract_bbox_from_mask(mask_data)

                    if bbox:
                        anomaly = AnomalyResult(
                            anomaly_id=f"{frame.frame_id}_{track.defect_type}_t{track.track_id}",
                            frame_id=frame.frame_id,
                            timestamp=frame.timestamp,
                            defect_type=track.defect_type,
                            structure_type=None,
                            bbox=bbox,
                            mask=SegmentationMask(data=mask_data, sam_score=track.confidences[-1]),
                            geometry=self._geometry.extract(mask_data),
                            detection_confidence=track.confidences[-1],
                            segmentation_confidence=track.confidences[-1],
                            association_confidence=1.0,
                        )
                        result.anomalies.append(anomaly)

            logger.info(f"[Frame 0] Initialized {len(tracks)} tracks, {len(result.anomalies)} anomalies")

        else:
            # Subsequent frames: Propagate existing tracks efficiently
            logger.info(f"[Frame {frame.frame_index}] Propagating tracks")

            frame_masks = self._video_tracker.propagate_tracks(
                frame.image,
                frame_idx=frame.frame_index
            )

            # Convert propagated tracks to anomalies
            for track_id, mask_data in frame_masks.items():
                track = self._video_tracker.get_track(track_id)
                if track:
                    bbox = self._extract_bbox_from_mask(mask_data)

                    if bbox:
                        anomaly = AnomalyResult(
                            anomaly_id=f"{frame.frame_id}_{track.defect_type}_t{track_id}",
                            frame_id=frame.frame_id,
                            timestamp=frame.timestamp,
                            defect_type=track.defect_type,
                            structure_type=None,
                            bbox=bbox,
                            mask=SegmentationMask(data=mask_data, sam_score=track.confidences[-1]),
                            geometry=self._geometry.extract(mask_data),
                            detection_confidence=track.confidences[-1],
                            segmentation_confidence=track.confidences[-1],
                            association_confidence=1.0,
                        )
                        result.anomalies.append(anomaly)

            logger.info(f"[Frame {frame.frame_index}] Propagated {len(frame_masks)} tracks, {len(result.anomalies)} anomalies")

        result.segmentation_time_ms = (time.perf_counter() - seg_start) * 1000
        result.sam_candidate_count = len(result.anomalies)
        result.vlm_time_ms = 0.0  # VLM disabled for video
        result.total_time_ms = (time.perf_counter() - start_time) * 1000

        # Debug: Save SAM3 summary
        self._debug.save_sam3_summary(
            frame.frame_id,
            frame.frame_index,
            len(result.anomalies),
            [a.defect_type for a in result.anomalies],
            result.segmentation_time_ms
        )

        return result

    async def _process_frame_fallback(
        self,
        frame: BufferedFrame,
        result: FrameResult,
        start_time: float
    ) -> FrameResult:
        """
        Fallback frame processing when video tracker is unavailable.

        Processes frames independently using class rotation (2 classes per frame).
        """
        seg_start = time.perf_counter()
        all_anomaly_classes = [cls.title() for cls in DEFAULT_ANOMALY_CLASSES]

        # Rotate: use 2 classes per frame, cycling through all classes
        num_classes = len(all_anomaly_classes)
        classes_per_frame = 2
        start_idx = (frame.frame_index * classes_per_frame) % num_classes
        anomaly_classes = [
            all_anomaly_classes[(start_idx + i) % num_classes]
            for i in range(classes_per_frame)
        ]

        logger.info(f"[Frame {frame.frame_index}] Running SAM3 on {len(anomaly_classes)} classes (rotating): {', '.join(anomaly_classes)}")

        sam_candidates = await self._segment_all_anomaly_classes(
            frame.image,
            frame.frame_id,
            anomaly_classes
        )
        result.segmentation_time_ms = (time.perf_counter() - seg_start) * 1000
        result.sam_candidate_count = len(sam_candidates)

        logger.info(f"[Frame {frame.frame_index}] SAM3 generated {len(sam_candidates)} candidate masks")

        # Debug: Save SAM3 summary
        self._debug.save_sam3_summary(
            frame.frame_id,
            frame.frame_index,
            len(sam_candidates),
            [c.defect_type for c in sam_candidates],
            result.segmentation_time_ms
        )

        # Convert SAM3 candidates directly to anomalies (NO VLM for video)
        for candidate in sam_candidates:
            anomaly = AnomalyResult(
                anomaly_id=f"{frame.frame_id}_{candidate.defect_type}",
                frame_id=frame.frame_id,
                timestamp=frame.timestamp,
                defect_type=candidate.defect_type,
                structure_type=None,
                bbox=candidate.bbox,
                mask=SegmentationMask(data=candidate.mask, sam_score=candidate.confidence),
                geometry=self._geometry.extract(candidate.mask),
                detection_confidence=candidate.confidence,
                segmentation_confidence=candidate.confidence,
                association_confidence=1.0,
            )
            result.anomalies.append(anomaly)

        result.vlm_time_ms = 0.0  # VLM disabled for video
        result.total_time_ms = (time.perf_counter() - start_time) * 1000

        return result

    def add_point_to_video_track(
        self,
        frame: np.ndarray,
        frame_idx: int,
        point_xy: tuple[int, int],
        defect_type: str,
        is_positive: bool = True,
    ) -> int | None:
        """
        Add a new object to track in the video via point prompt.

        This allows frame-level refinement of video tracking by adding point prompts
        on specific frames during video processing.

        Args:
            frame: RGB image (HWC)
            frame_idx: Frame index
            point_xy: Point coordinates (x, y)
            defect_type: Type of defect to track
            is_positive: True for positive point (foreground), False for negative

        Returns:
            Track ID if successful, None otherwise
        """
        if not self._video_tracker or not self._video_tracker.is_loaded():
            logger.warning("Video tracker not available, cannot add point prompt")
            return None

        try:
            track = self._video_tracker.add_point_prompt(
                frame=frame,
                frame_idx=frame_idx,
                point_xy=point_xy,
                defect_type=defect_type,
                is_positive=is_positive,
            )

            if track:
                logger.info(f"Added track {track.track_id} via point prompt at {point_xy} on frame {frame_idx}")
                return track.track_id
            else:
                logger.warning(f"Failed to add track via point prompt at {point_xy} on frame {frame_idx}")
                return None

        except Exception as e:
            logger.error(f"Error adding point prompt to video track: {e}")
            return None

    def get_video_tracks(self) -> list:
        """
        Get all active video tracks.

        Returns:
            List of TrackingState objects representing active tracks
        """
        if not self._video_tracker or not self._video_tracker.is_loaded():
            return []

        return self._video_tracker.get_active_tracks()

    def deactivate_video_track(self, track_id: int) -> None:
        """
        Deactivate a specific video track.

        Args:
            track_id: ID of track to deactivate
        """
        if self._video_tracker and self._video_tracker.is_loaded():
            self._video_tracker.deactivate_track(track_id)

    async def _segment_all_anomaly_classes(
        self,
        image: np.ndarray,
        frame_id: str,
        anomaly_classes: list[str],
    ) -> list[SAMCandidate]:
        """
        Segment image using ALL anomaly classes as text prompts for SAM3.

        Args:
            image: RGB image
            frame_id: Frame identifier
            anomaly_classes: List of defect types in Title Case (Crack, Corrosion, etc.)

        Returns:
            List of SAM candidate masks
        """
        candidates = []
        self._segmenter.set_image(image)

        try:
            for defect_type in anomaly_classes:
                # Create fake detection with defect type as text prompt
                fake_detection = Detection(
                    detection_id=f"{frame_id}_{defect_type}",
                    detection_type=Detection,
                    class_name=defect_type,  # SAM3 uses this as text prompt
                    confidence=1.0,
                    bbox=BoundingBox(0, 0, image.shape[1], image.shape[0]),  # Full image
                )

                # Run SAM3 segmentation
                result = self._segmenter.segment_detection(fake_detection, image=None)

                if result.success and result.mask and result.mask.data is not None:
                    # Extract bounding box from mask
                    bbox = self._extract_bbox_from_mask(result.mask.data)

                    if bbox:
                        candidate = SAMCandidate(
                            defect_type=defect_type,
                            mask=result.mask.data,
                            bbox=bbox,
                            confidence=result.mask.sam_score if hasattr(result.mask, 'sam_score') else 0.5,
                        )
                        candidates.append(candidate)
                        logger.debug(f"  SAM3 found {defect_type}: bbox={bbox}")

                        # Debug: Save SAM3 candidate
                        self._debug.save_sam3_candidate(
                            frame_id,
                            int(frame_id.split('_')[-1]) if '_' in frame_id else 0,
                            defect_type,
                            result.mask.data,
                            image,
                            candidate.confidence
                        )

        finally:
            self._segmenter.clear_image()

        return candidates

    def _extract_bbox_from_mask(self, mask: np.ndarray) -> BoundingBox | None:
        """Extract bounding box from binary mask."""
        import cv2

        if mask is None or mask.sum() == 0:
            return None

        # Find contours
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Get bounding box from largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        return BoundingBox(x, y, x + w, y + h)

    def _filter_by_vlm_judgment(
        self,
        sam_candidates: list[SAMCandidate],
        vlm_predictions: list[VLMPrediction],
        frame_id: str,
    ) -> list[AnomalyResult]:
        """
        Filter SAM3 candidates by VLM judgment.

        Args:
            sam_candidates: SAM3-generated candidate masks
            vlm_predictions: VLM judgment predictions
            frame_id: Frame identifier

        Returns:
            List of approved anomaly results
        """
        approved = []

        # Match VLM predictions to SAM candidates by defect type
        for vlm_pred in vlm_predictions:
            if vlm_pred.confidence < self.config.confidence_threshold:
                continue  # VLM rejected this candidate

            # Find matching SAM candidate
            matching_candidate = None
            for candidate in sam_candidates:
                if candidate.defect_type.lower() == vlm_pred.defect_type.lower():
                    matching_candidate = candidate
                    break

            if matching_candidate:
                # Extract geometry
                geometry = self._geometry.extract(matching_candidate.mask)

                # Build approved anomaly result
                anomaly = AnomalyResult(
                    anomaly_id=f"vlm_approved_{frame_id}_{vlm_pred.defect_type}",
                    frame_id=frame_id,
                    timestamp=time.time(),
                    defect_type=vlm_pred.defect_type,
                    structure_type=None,
                    bbox=matching_candidate.bbox,
                    mask=SegmentationMask(data=matching_candidate.mask),
                    geometry=geometry,
                    detection_confidence=vlm_pred.confidence,  # VLM confidence
                    segmentation_confidence=matching_candidate.confidence,  # SAM3 confidence
                    association_confidence=1.0,  # Perfect match by defect type
                )
                approved.append(anomaly)

        return approved

    def stop(self) -> None:
        """Stop processing."""
        self._is_running = False
        if self._stream:
            self._stream.stop()

    def set_on_result_callback(self, callback: Callable[[FrameResult], None]) -> None:
        """Set callback for frame results."""
        self._on_result_callback = callback

    def get_stats(self) -> ProcessingStats:
        """Get processing statistics."""
        if self._processing_times:
            self._stats.avg_total_frame_time_ms = sum(self._processing_times) / len(self._processing_times)
            if self._stats.avg_total_frame_time_ms > 0:
                self._stats.avg_fps = 1000.0 / self._stats.avg_total_frame_time_ms

        if self._vlm_judge:
            self._stats.vlm_stats = self._vlm_judge.get_stats()

        return self._stats

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def current_frame_index(self) -> int:
        return self._current_frame_index

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unload()

    async def __aenter__(self):
        self.load()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._vlm_judge:
            await self._vlm_judge.cancel_pending()
        self.unload()
