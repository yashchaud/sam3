"""Real-time video processor with VLM-guided segmentation."""

import asyncio
import time
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
from ..detector import RFDETRDetector, DetectorConfig
from ..segmenter import SAM3Segmenter, SegmenterConfig
from ..models import Detection, AnomalyResult, PipelineOutput, BoundingBox, SegmentationMask
from ..tiling import TiledDetectionCoordinator, TileConfig
from ..association import StructureDefectMatcher
from ..geometry import MaskGeometryExtractor


@dataclass
class FrameResult:
    """Result from processing a single frame."""
    frame_id: str
    frame_index: int
    timestamp: float

    # Detection results
    anomalies: list[AnomalyResult] = field(default_factory=list)
    structures: list[Detection] = field(default_factory=list)

    # VLM-guided additions
    vlm_guided_anomalies: list[AnomalyResult] = field(default_factory=list)
    vlm_response: VLMResponse | None = None

    # Timing
    detection_time_ms: float = 0.0
    segmentation_time_ms: float = 0.0
    vlm_time_ms: float = 0.0
    total_time_ms: float = 0.0

    @property
    def all_anomalies(self) -> list[AnomalyResult]:
        return self.anomalies + self.vlm_guided_anomalies

    def to_dict(self) -> dict:
        return {
            "frame_id": self.frame_id,
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
            "anomaly_count": len(self.all_anomalies),
            "structure_count": len(self.structures),
            "vlm_guided_count": len(self.vlm_guided_anomalies),
            "timing": {
                "detection_ms": round(self.detection_time_ms, 2),
                "segmentation_ms": round(self.segmentation_time_ms, 2),
                "vlm_ms": round(self.vlm_time_ms, 2),
                "total_ms": round(self.total_time_ms, 2),
            },
        }


class RealtimeVideoProcessor:
    """
    Real-time video processor with VLM-guided anomaly detection.

    Pipeline:
    1. Stream frames from video/webcam/RTSP
    2. Run RF-DETR detection on each frame
    3. Run SAM3 segmentation on detected anomalies
    4. Every N frames, send to VLM for additional guidance
    5. VLM predictions are processed async and merged when ready
    6. Stale VLM predictions (>60 frames old) are discarded
    """

    def __init__(self, config: RealtimeConfig):
        self.config = config

        # Initialize components
        self._detector: RFDETRDetector | None = None
        self._segmenter: SAM3Segmenter | None = None
        self._vlm_judge: VLMJudge | None = None
        self._matcher = StructureDefectMatcher()
        self._geometry = MaskGeometryExtractor()

        # Tiling coordinator for large frames
        self._tiled_coordinator: TiledDetectionCoordinator | None = None

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

        # VLM pending predictions
        self._pending_vlm_predictions: list[VLMPrediction] = []

        # Result callbacks
        self._on_result_callback: Callable[[FrameResult], None] | None = None

    def load(self) -> None:
        """Load all models into memory."""
        if self._is_loaded:
            return

        # Initialize detector
        detector_config = self.config.get_detector_config()
        self._detector = RFDETRDetector(detector_config)
        self._detector.load()

        # Initialize segmenter
        segmenter_config = self.config.get_segmenter_config()
        self._segmenter = SAM3Segmenter(segmenter_config)
        self._segmenter.load()

        # Initialize tiled coordinator if needed
        if self.config.enable_tiling:
            self._tiled_coordinator = TiledDetectionCoordinator(
                detector=self._detector,
                segmenter=self._segmenter,
                tile_config=self.config.tile_config,
            )

        # Initialize VLM judge if enabled
        if self.config.enable_vlm_judge:
            vlm_config = self.config.get_vlm_config()
            self._vlm_judge = VLMJudge(vlm_config)
            self._vlm_judge.load()

        self._is_loaded = True

    def unload(self) -> None:
        """Unload all models from memory."""
        if self._vlm_judge:
            self._vlm_judge.unload()
            self._vlm_judge = None

        if self._segmenter:
            self._segmenter.unload()
            self._segmenter = None

        if self._detector:
            self._detector.unload()
            self._detector = None

        self._tiled_coordinator = None
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
        Process a single image.

        Args:
            image: RGB image (HWC)
            frame_id: Optional frame identifier

        Returns:
            FrameResult with all detections
        """
        if not self._is_loaded:
            raise RuntimeError("Models not loaded. Call load() first.")

        buffered = BufferedFrame(
            image=image,
            frame_index=self._current_frame_index,
            timestamp=time.time(),
            frame_id=frame_id or f"image_{self._current_frame_index:08d}",
        )

        result = await self._process_frame(buffered)
        self._current_frame_index += 1

        return result

    async def _process_frame(self, frame: BufferedFrame) -> FrameResult:
        """Process a single buffered frame."""
        start_time = time.perf_counter()

        # Initialize result
        result = FrameResult(
            frame_id=frame.frame_id,
            frame_index=frame.frame_index,
            timestamp=frame.timestamp,
        )

        # Step 1: Run detection
        detection_start = time.perf_counter()
        detections = self._run_detection(frame.image)
        result.detection_time_ms = (time.perf_counter() - detection_start) * 1000

        # Separate structures and anomalies
        structures = [d for d in detections if d.detection_type.value == "structure"]
        anomalies = [d for d in detections if d.detection_type.value == "anomaly"]
        result.structures = structures

        # Step 2: Run segmentation on anomalies
        seg_start = time.perf_counter()
        segmented_anomalies = self._run_segmentation(frame.image, anomalies, structures)
        result.segmentation_time_ms = (time.perf_counter() - seg_start) * 1000
        result.anomalies = segmented_anomalies

        # Step 3: Check for VLM-guided predictions
        vlm_start = time.perf_counter()
        if self._vlm_judge and self.config.enable_vlm_judge:
            # Get any ready VLM predictions
            ready_predictions = await self._vlm_judge.get_ready_predictions(
                frame.frame_index
            )

            for response in ready_predictions:
                result.vlm_response = response
                self._stats.total_vlm_predictions += len(response.predictions)

                # Process VLM predictions
                vlm_anomalies = await self._process_vlm_predictions(
                    frame.image,
                    response.predictions,
                    frame.frame_id,
                )
                result.vlm_guided_anomalies.extend(vlm_anomalies)

            # Submit new frame for VLM processing if appropriate
            if self._vlm_judge.should_process_frame(frame.frame_index):
                await self._vlm_judge.submit_frame_async(
                    frame.image,
                    frame.frame_id,
                    frame.frame_index,
                )

        result.vlm_time_ms = (time.perf_counter() - vlm_start) * 1000
        result.total_time_ms = (time.perf_counter() - start_time) * 1000

        return result

    def _run_detection(self, image: np.ndarray) -> list[Detection]:
        """Run object detection on image."""
        if self._tiled_coordinator and self.config.enable_tiling:
            # Use tiled detection for large images
            tiled_results = self._tiled_coordinator.detect_only(image)
            return [r.detection for r in tiled_results]
        else:
            # Direct detection
            output = self._detector.detect(image)
            return output.detections

    def _run_segmentation(
        self,
        image: np.ndarray,
        anomalies: list[Detection],
        structures: list[Detection],
    ) -> list[AnomalyResult]:
        """Run segmentation on detected anomalies."""
        if not anomalies:
            return []

        results = []

        # Set image for batch processing
        self._segmenter.set_image(image)

        try:
            for detection in anomalies:
                # Segment this detection
                seg_result = self._segmenter.segment_detection(detection)

                if not seg_result.success or seg_result.mask is None:
                    continue

                # Extract geometry
                geometry = self._geometry.extract(seg_result.mask.data)

                # Match to structure
                match_results = self._matcher.match([detection], structures)
                match = match_results[0] if match_results else None

                # Build anomaly result
                anomaly = AnomalyResult(
                    anomaly_id=detection.detection_id,
                    frame_id="",  # Will be set by caller
                    timestamp=time.time(),
                    defect_type=detection.class_name,
                    structure_type=match.structure_class if match else None,
                    bbox=detection.bbox,
                    mask=seg_result.mask,
                    geometry=geometry,
                    detection_confidence=detection.confidence,
                    segmentation_confidence=seg_result.mask.sam_score or 0.0,
                    association_confidence=match.confidence if match else 0.0,
                )
                results.append(anomaly)

        finally:
            self._segmenter.clear_image()

        return results

    async def _process_vlm_predictions(
        self,
        image: np.ndarray,
        predictions: list[VLMPrediction],
        frame_id: str,
    ) -> list[AnomalyResult]:
        """Process VLM predictions and run segmentation."""
        if not predictions:
            return []

        results = []
        self._segmenter.set_image(image)

        try:
            for pred in predictions:
                # Create detection-like prompt for SAM
                if pred.prediction_type == PredictionType.BOX and pred.box:
                    # Use box prompt
                    bbox = BoundingBox(*pred.box)
                    mask_data = self._segment_with_box(bbox)
                elif pred.prediction_type == PredictionType.POINT and pred.point:
                    # Use point prompt
                    mask_data = self._segment_with_point(pred.point)
                else:
                    continue

                if mask_data is None:
                    continue

                # Extract geometry
                geometry = self._geometry.extract(mask_data)

                # Build anomaly result
                x_min = int(pred.box[0]) if pred.box else pred.point[0] - 50
                y_min = int(pred.box[1]) if pred.box else pred.point[1] - 50
                x_max = int(pred.box[2]) if pred.box else pred.point[0] + 50
                y_max = int(pred.box[3]) if pred.box else pred.point[1] + 50

                anomaly = AnomalyResult(
                    anomaly_id=f"vlm_{frame_id}_{len(results)}",
                    frame_id=frame_id,
                    timestamp=time.time(),
                    defect_type=pred.defect_type,
                    structure_type=None,
                    bbox=BoundingBox(x_min, y_min, x_max, y_max),
                    mask=SegmentationMask(data=mask_data),
                    geometry=geometry,
                    detection_confidence=pred.confidence,
                    segmentation_confidence=0.8,  # VLM-guided
                    association_confidence=0.0,
                )
                results.append(anomaly)

        finally:
            self._segmenter.clear_image()

        return results

    def _segment_with_box(self, bbox: BoundingBox) -> np.ndarray | None:
        """Run SAM segmentation with box prompt."""
        try:
            # Create fake detection for segmentation
            fake_detection = Detection(
                detection_id="vlm_box",
                detection_type=Detection,  # Will be ignored
                class_name="vlm_prediction",
                confidence=0.9,
                bbox=bbox,
            )
            result = self._segmenter.segment_detection(fake_detection)
            if result.success and result.mask:
                return result.mask.data
        except Exception:
            pass
        return None

    def _segment_with_point(self, point: tuple[int, int]) -> np.ndarray | None:
        """Run SAM segmentation with point prompt."""
        try:
            # Use SAM's point prompting
            import torch

            if not hasattr(self._segmenter, '_model') or self._segmenter._model is None:
                return None

            input_points = torch.tensor([[[point[0], point[1]]]], dtype=torch.float32)
            input_labels = torch.tensor([[[1]]], dtype=torch.int32)

            if self._segmenter._device == "cuda":
                input_points = input_points.cuda()
                input_labels = input_labels.cuda()

            # Run prediction
            with torch.no_grad():
                masks, scores, _ = self._segmenter._model.predict(
                    input_points=input_points,
                    input_labels=input_labels,
                    multimask_output=True,
                )

            # Select best mask
            best_idx = scores.argmax().item()
            mask = masks[0, best_idx].cpu().numpy()

            return (mask > 0.5).astype(np.uint8)

        except Exception:
            pass
        return None

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
