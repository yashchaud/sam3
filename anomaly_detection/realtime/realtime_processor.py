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
    ) -> FrameResult:
        """
        Process a single image synchronously (waits for VLM response).

        For single images, we wait for VLM to respond and include SAM3 candidates
        in the result even if VLM doesn't approve them (shown with lower confidence).

        Args:
            image: RGB image (HWC)
            frame_id: Optional frame identifier

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

        # Process with synchronous VLM call for single images
        result = await self._process_single_image_sync(buffered)
        self._current_frame_index += 1

        return result

    async def _process_single_image_sync(self, frame: BufferedFrame) -> FrameResult:
        """
        Process a single image with synchronous VLM call.

        Unlike video processing, this waits for VLM response and includes
        all SAM3 candidates in the output (VLM-approved ones marked accordingly).
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

        # STEP 1: Run SAM3 on ALL default anomaly classes
        seg_start = time.perf_counter()
        anomaly_classes = [cls.title() for cls in DEFAULT_ANOMALY_CLASSES]

        logger.info(f"[Frame {frame.frame_index}] Running SAM3 on {len(anomaly_classes)} anomaly classes")

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

        # STEP 2: Call VLM synchronously (wait for response)
        vlm_start = time.perf_counter()
        vlm_response = None

        if sam_candidates:
            logger.info(f"[Frame {frame.frame_index}] Calling VLM judge synchronously...")

            # Generate grid overlay
            self._vlm_judge.grid.compute_grid(frame.image.shape[1], frame.image.shape[0])
            grid_image = self._vlm_judge.grid.draw_grid(frame.image)

            # Debug: Save VLM grid overlay
            self._debug.save_vlm_grid(frame.frame_id, frame.frame_index, grid_image)

            # Call VLM synchronously (not async submit)
            vlm_response = await self._vlm_judge.process_frame(
                frame.image,
                frame.frame_id,
                frame.frame_index,
            )

            if vlm_response:
                result.vlm_response = vlm_response
                logger.info(f"[Frame {frame.frame_index}] VLM response: {len(vlm_response.predictions)} predictions, valid={vlm_response.is_valid}")

                if vlm_response.error_message:
                    logger.warning(f"[Frame {frame.frame_index}] VLM error: {vlm_response.error_message}")

                # Debug: Save VLM response
                self._debug.save_vlm_response(
                    frame.frame_id,
                    frame.frame_index,
                    vlm_response.predictions,
                    vlm_response.generation_time_ms,
                    frame.frame_index
                )

        result.vlm_time_ms = (time.perf_counter() - vlm_start) * 1000

        # STEP 3: Convert SAM3 candidates to anomaly results
        # Include ALL candidates, but mark VLM-approved ones with higher confidence
        for candidate in sam_candidates:
            # Check if VLM approved this defect type
            vlm_confidence = 0.0
            if vlm_response and vlm_response.predictions:
                for pred in vlm_response.predictions:
                    if pred.defect_type.lower() == candidate.defect_type.lower():
                        vlm_confidence = pred.confidence
                        break

            # If VLM approved (confidence >= threshold), add to vlm_judged_anomalies
            # Otherwise add to anomalies (SAM3 candidates without VLM approval)
            geometry = self._geometry.extract(candidate.mask)

            anomaly = AnomalyResult(
                anomaly_id=f"{frame.frame_id}_{candidate.defect_type}",
                frame_id=frame.frame_id,
                timestamp=time.time(),
                defect_type=candidate.defect_type,
                structure_type=None,
                bbox=candidate.bbox,
                mask=SegmentationMask(data=candidate.mask, sam_score=candidate.confidence),
                geometry=geometry,
                detection_confidence=vlm_confidence if vlm_confidence > 0 else candidate.confidence,
                segmentation_confidence=candidate.confidence,
                association_confidence=1.0 if vlm_confidence > 0 else 0.5,
            )

            if vlm_confidence >= self.config.confidence_threshold:
                result.vlm_judged_anomalies.append(anomaly)
                logger.info(f"[Frame {frame.frame_index}] ✓ VLM approved: {candidate.defect_type} (conf={vlm_confidence:.2f})")
            else:
                result.anomalies.append(anomaly)
                logger.info(f"[Frame {frame.frame_index}] SAM3 candidate (no VLM approval): {candidate.defect_type}")

        # Debug: Save approved detections
        if result.all_anomalies:
            self._debug.save_approved_detections(
                frame.frame_id,
                frame.frame_index,
                frame.image,
                result.all_anomalies,
                len(sam_candidates)
            )

        result.total_time_ms = (time.perf_counter() - start_time) * 1000

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

        logger.info(f"[Frame {frame.frame_index}] Complete: {len(result.all_anomalies)} anomalies "
                    f"({len(result.vlm_judged_anomalies)} VLM-approved, {len(result.anomalies)} SAM3-only)")

        return result

    async def _process_frame(self, frame: BufferedFrame) -> FrameResult:
        """
        Process a single buffered frame using SAM3-first pipeline.

        Pipeline:
        1. SAM3 segments ALL default anomaly classes (Title Case as text prompts)
        2. Submit SAM3 candidates to VLM for judging (async)
        3. Check for ready VLM judgments from previous frames
        4. Only keep VLM-approved detections
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

        # STEP 1: Run SAM3 on ALL default anomaly classes (Title Case - first letter caps)
        seg_start = time.perf_counter()
        anomaly_classes = [cls.title() for cls in DEFAULT_ANOMALY_CLASSES]  # Convert to Title Case

        logger.info(f"[Frame {frame.frame_index}] Running SAM3 on {len(anomaly_classes)} anomaly classes: {', '.join(anomaly_classes)}")

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

        # Store candidates for this frame
        self._pending_sam_candidates[frame.frame_id] = sam_candidates

        # STEP 2: Submit to VLM for judging (async) - every N frames
        vlm_start = time.perf_counter()

        if self._vlm_judge.should_process_frame(frame.frame_index):
            logger.info(f"[Frame {frame.frame_index}] Submitting frame to VLM judge (will judge {len(sam_candidates)} SAM3 candidates)")

            # Generate grid overlay for debug output
            self._vlm_judge.grid.compute_grid(frame.image.shape[1], frame.image.shape[0])
            grid_image = self._vlm_judge.grid.draw_grid(frame.image)

            # Debug: Save VLM grid overlay
            self._debug.save_vlm_grid(frame.frame_id, frame.frame_index, grid_image)

            await self._vlm_judge.submit_frame_async(
                frame.image,
                frame.frame_id,
                frame.frame_index,
            )

        # STEP 3: Check for ready VLM judgments from previous frames
        ready_judgments = await self._vlm_judge.get_ready_predictions(
            frame.frame_index
        )

        for response in ready_judgments:
            result.vlm_response = response
            self._stats.total_vlm_predictions += len(response.predictions)

            # Debug: Save VLM response
            self._debug.save_vlm_response(
                response.frame_id,
                frame.frame_index,
                response.predictions,
                response.generation_time_ms,
                response.request_frame_index
            )

            # Log VLM judgments
            if response.predictions:
                approved_count = sum(1 for p in response.predictions if p.confidence >= self.config.confidence_threshold)
                logger.info(
                    f"[Frame {response.frame_id}] VLM JUDGE: approved {approved_count}/{len(response.predictions)} "
                    f"SAM3 candidates (latency: {response.generation_time_ms:.0f}ms)"
                )
                for pred in response.predictions:
                    verdict = "✓ APPROVED" if pred.confidence >= self.config.confidence_threshold else "✗ REJECTED"
                    logger.info(
                        f"  {verdict}: {pred.defect_type.title()}, confidence: {pred.confidence:.2f}"
                    )

            # STEP 4: Filter SAM3 candidates by VLM judgment
            if response.frame_id in self._pending_sam_candidates:
                sam_candidates_for_frame = self._pending_sam_candidates[response.frame_id]

                approved_anomalies = self._filter_by_vlm_judgment(
                    sam_candidates_for_frame,
                    response.predictions,
                    response.frame_id,
                )
                result.vlm_judged_anomalies.extend(approved_anomalies)

                # Debug: Save approved detections
                self._debug.save_approved_detections(
                    response.frame_id,
                    frame.frame_index,
                    frame.image,
                    approved_anomalies,
                    len(sam_candidates_for_frame)
                )

                # Clean up processed candidates
                del self._pending_sam_candidates[response.frame_id]

        result.vlm_time_ms = (time.perf_counter() - vlm_start) * 1000
        result.total_time_ms = (time.perf_counter() - start_time) * 1000

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
