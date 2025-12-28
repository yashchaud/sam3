"""VLM Judge - Unified interface for VLM-based anomaly detection guidance."""

import asyncio
import time
import numpy as np
from typing import AsyncGenerator
from collections import deque
from dataclasses import dataclass

from .models import (
    VLMConfig,
    VLMPrediction,
    VLMResponse,
    VLMProvider,
    LatencyStats,
    PredictionType,
)
from .grid_overlay import GridOverlay
from .qwen_client import QwenVLClient
from .openrouter_client import OpenRouterClient
from .base_client import BaseVLMClient


@dataclass
class PendingRequest:
    """Tracks a pending VLM request."""
    frame_id: str
    request_frame_index: int
    task: asyncio.Task
    timestamp: float


class VLMJudge:
    """
    Unified VLM Judge for anomaly detection guidance.

    Handles:
    - Grid overlay on images
    - Async VLM processing with timeout
    - Automatic discard of stale predictions
    - Latency statistics tracking
    - Support for both local and cloud providers
    """

    def __init__(self, config: VLMConfig):
        self.config = config
        self.grid = GridOverlay(config.grid_config)
        self.stats = LatencyStats()

        # Initialize client based on provider
        self._client: BaseVLMClient
        if config.provider == VLMProvider.LOCAL_QWEN:
            self._client = QwenVLClient(config)
        else:
            self._client = OpenRouterClient(config)

        # Pending requests for async processing
        self._pending: deque[PendingRequest] = deque(maxlen=10)
        self._current_frame_index = 0
        self._lock = asyncio.Lock()

    def load(self) -> None:
        """Load the VLM model."""
        self._client.load()

    def unload(self) -> None:
        """Unload the VLM model."""
        self._client.unload()
        if isinstance(self._client, OpenRouterClient):
            # Schedule session close
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(self._client.close())
                else:
                    loop.run_until_complete(self._client.close())
            except RuntimeError:
                pass

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._client.is_loaded()

    async def process_frame(
        self,
        image: np.ndarray,
        frame_id: str,
        frame_index: int,
    ) -> VLMResponse | None:
        """
        Process a single frame synchronously.

        Args:
            image: RGB image (HWC)
            frame_id: Unique frame identifier
            frame_index: Current frame index in video

        Returns:
            VLMResponse if successful, None if skipped
        """
        self._current_frame_index = frame_index

        # Compute grid for this image
        self.grid.compute_grid(image.shape[1], image.shape[0])

        # Create grid overlay image
        grid_image = self.grid.draw_grid(image)

        # Get grid description
        grid_description = self.grid.get_grid_description()

        # Run prediction
        response = await self._client.predict(
            image=image,
            grid_image=grid_image,
            grid_description=grid_description,
            frame_id=frame_id,
        )

        # Resolve grid cells to actual coordinates
        response = self._resolve_coordinates(response, image.shape[1], image.shape[0])

        # Record stats
        self.stats.record_latency(
            response.generation_time_ms,
            success=response.is_valid,
            discarded=False,
        )

        return response

    async def submit_frame_async(
        self,
        image: np.ndarray,
        frame_id: str,
        frame_index: int,
    ) -> None:
        """
        Submit a frame for async processing.

        The result can be retrieved later using get_ready_predictions().
        Predictions are automatically discarded if they take too long.

        Args:
            image: RGB image (HWC)
            frame_id: Unique frame identifier
            frame_index: Current frame index in video
        """
        self._current_frame_index = frame_index

        # Compute grid
        self.grid.compute_grid(image.shape[1], image.shape[0])
        grid_image = self.grid.draw_grid(image)
        grid_description = self.grid.get_grid_description()

        # Create async task
        async def _run():
            response = await self._client.predict(
                image=image,
                grid_image=grid_image,
                grid_description=grid_description,
                frame_id=frame_id,
            )
            response.request_frame_index = frame_index
            return self._resolve_coordinates(response, image.shape[1], image.shape[0])

        task = asyncio.create_task(_run())

        async with self._lock:
            self._pending.append(PendingRequest(
                frame_id=frame_id,
                request_frame_index=frame_index,
                task=task,
                timestamp=time.time(),
            ))

    async def get_ready_predictions(
        self,
        current_frame_index: int,
    ) -> list[VLMResponse]:
        """
        Get all ready predictions that are still valid.

        Discards predictions that took longer than max_generation_frames.

        Args:
            current_frame_index: Current frame index for staleness check

        Returns:
            List of valid VLMResponse objects
        """
        self._current_frame_index = current_frame_index
        ready_responses = []

        async with self._lock:
            completed = []
            remaining = []

            for request in self._pending:
                if request.task.done():
                    completed.append(request)
                else:
                    remaining.append(request)

            self._pending = deque(remaining, maxlen=10)

            for request in completed:
                try:
                    response = request.task.result()
                    response.current_frame_index = current_frame_index

                    # Check if response is stale
                    frame_delay = current_frame_index - request.request_frame_index
                    is_stale = frame_delay > self.config.max_generation_frames

                    self.stats.record_latency(
                        response.generation_time_ms,
                        success=response.is_valid,
                        discarded=is_stale,
                    )

                    if not is_stale and response.is_valid:
                        ready_responses.append(response)

                except Exception as e:
                    self.stats.record_latency(0, success=False, discarded=False)

        return ready_responses

    async def cancel_pending(self) -> None:
        """Cancel all pending requests."""
        async with self._lock:
            for request in self._pending:
                if not request.task.done():
                    request.task.cancel()
            self._pending.clear()

    def should_process_frame(self, frame_index: int) -> bool:
        """Check if this frame should be processed by VLM."""
        return frame_index % self.config.process_every_n_frames == 0

    def _resolve_coordinates(
        self,
        response: VLMResponse,
        image_width: int,
        image_height: int,
    ) -> VLMResponse:
        """Resolve grid cell references to actual pixel coordinates."""
        resolved_predictions = []

        for pred in response.predictions:
            if pred.grid_cell is None:
                continue

            col, row = pred.grid_cell
            cell = self.grid.get_cell(col, row)

            if cell is None:
                continue

            if self.config.prefer_boxes:
                # Use cell bounds as box
                resolved_predictions.append(VLMPrediction(
                    prediction_type=PredictionType.BOX,
                    confidence=pred.confidence,
                    defect_type=pred.defect_type,
                    box=cell.bounds,
                    grid_cell=pred.grid_cell,
                ))
            else:
                # Use cell center as point
                resolved_predictions.append(VLMPrediction(
                    prediction_type=PredictionType.POINT,
                    confidence=pred.confidence,
                    defect_type=pred.defect_type,
                    point=cell.center,
                    grid_cell=pred.grid_cell,
                ))

        return VLMResponse(
            frame_id=response.frame_id,
            predictions=resolved_predictions,
            generation_time_ms=response.generation_time_ms,
            is_valid=response.is_valid,
            error_message=response.error_message,
            provider=response.provider,
            request_frame_index=response.request_frame_index,
            current_frame_index=response.current_frame_index,
        )

    def get_stats(self) -> dict:
        """Get latency statistics."""
        return self.stats.to_dict()

    def reset_stats(self) -> None:
        """Reset latency statistics."""
        self.stats.reset()

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unload()

    async def __aenter__(self):
        self.load()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cancel_pending()
        self.unload()
