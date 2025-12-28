"""SAM3 Video Tracker for efficient video segmentation with temporal consistency."""

import logging
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

from .sam3_segmenter import SegmenterConfig
from ..models import SegmentationMask

logger = logging.getLogger(__name__)


@dataclass
class TrackingState:
    """State for tracking objects across video frames."""
    track_id: int
    defect_type: str
    masks: List[np.ndarray]  # Mask for each frame
    confidences: List[float]
    frame_indices: List[int]
    is_active: bool = True


class SAM3VideoTracker:
    """
    Video tracker using Sam3TrackerModel for efficient temporal segmentation.

    Features:
    - Track objects across frames with temporal consistency
    - Add point/box prompts on any frame for refinement
    - Propagate masks forward through video
    - Much faster than per-frame processing
    """

    def __init__(self, config: SegmenterConfig):
        self.config = config
        self._device = config.device
        self._model = None
        self._processor = None
        self._is_loaded = False

        # Tracking state
        self._tracks: Dict[int, TrackingState] = {}
        self._next_track_id = 0
        self._current_frame_idx = 0

    def load(self) -> None:
        """Load Sam3TrackerModel."""
        if self._is_loaded:
            return

        try:
            import torch

            # Determine device (resolve "auto" to actual device)
            if self.config.device == "auto":
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self._device = self.config.device

            logger.info(f"Loading SAM3 Tracker model on device: {self._device}")

            from transformers import Sam3TrackerModel, Sam3Processor

            model_kwargs = {}
            if self.config.hf_token:
                model_kwargs["token"] = self.config.hf_token

            self._model = Sam3TrackerModel.from_pretrained("facebook/sam3", **model_kwargs).to(self._device)
            self._processor = Sam3Processor.from_pretrained("facebook/sam3", **model_kwargs)

            self._is_loaded = True
            logger.info("SAM3 Tracker model loaded successfully")

        except ImportError as e:
            raise ImportError(
                f"SAM3TrackerModel not available. Install transformers with SAM3 support.\nError: {e}"
            )

    def unload(self) -> None:
        """Unload model from memory."""
        if self._model is not None:
            del self._model
            self._model = None

        if self._processor is not None:
            del self._processor
            self._processor = None

        self._tracks.clear()
        self._is_loaded = False

    def is_loaded(self) -> bool:
        return self._is_loaded

    def reset_tracking(self) -> None:
        """Reset all tracking state (call at start of new video)."""
        self._tracks.clear()
        self._next_track_id = 0
        self._current_frame_idx = 0
        logger.info("Video tracking state reset")

    def initialize_with_text_prompts(
        self,
        frame: np.ndarray,
        text_prompts: List[str],
        frame_idx: int = 0,
    ) -> List[TrackingState]:
        """
        Initialize tracking on first frame using text prompts.

        Args:
            frame: RGB image (HWC)
            text_prompts: List of defect types to detect
            frame_idx: Frame index

        Returns:
            List of initialized tracks
        """
        if not self.is_loaded():
            raise RuntimeError("Tracker not loaded. Call load() first.")

        import torch
        from PIL import Image

        logger.info(f"[Tracker] Initializing with {len(text_prompts)} text prompts on frame {frame_idx}")

        # Convert to PIL
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
        frame_pil = Image.fromarray(frame)

        new_tracks = []

        # Process each text prompt
        for defect_type in text_prompts:
            try:
                # Prepare inputs
                inputs = self._processor(
                    images=frame_pil,
                    text=defect_type,
                    return_tensors="pt"
                ).to(self._device)

                # Run model
                with torch.no_grad():
                    outputs = self._model(**inputs)

                # Post-process with lower thresholds for text-based detection
                # Text prompts are less precise than point/box, so we use more lenient thresholds
                text_score_threshold = min(self.config.score_threshold, 0.15)
                text_mask_threshold = min(self.config.mask_threshold, 0.3)

                results = self._processor.post_process_instance_segmentation(
                    outputs,
                    threshold=text_score_threshold,
                    mask_threshold=text_mask_threshold,
                    target_sizes=[[frame.shape[0], frame.shape[1]]]
                )[0]

                # Create tracks for each detected mask
                num_masks = len(results.get('masks', []))
                if num_masks > 0:
                    for i, mask_tensor in enumerate(results['masks']):
                        # Extract mask
                        if hasattr(mask_tensor, 'cpu'):
                            mask_data = mask_tensor.cpu().numpy()
                        else:
                            mask_data = np.array(mask_tensor)

                        if mask_data.ndim > 2:
                            mask_data = mask_data.squeeze()
                        mask_binary = (mask_data > 0).astype(np.uint8)

                        if mask_binary.sum() == 0:
                            continue

                        # Get confidence
                        confidence = 1.0
                        if len(results.get('scores', [])) > i:
                            score_tensor = results['scores'][i]
                            if hasattr(score_tensor, 'cpu'):
                                confidence = float(score_tensor.cpu().numpy())
                            else:
                                confidence = float(score_tensor)

                        # Create tracking state
                        track = TrackingState(
                            track_id=self._next_track_id,
                            defect_type=defect_type,
                            masks=[mask_binary],
                            confidences=[confidence],
                            frame_indices=[frame_idx],
                            is_active=True,
                        )

                        self._tracks[self._next_track_id] = track
                        new_tracks.append(track)
                        self._next_track_id += 1

                        logger.info(f"[Tracker] Created track {track.track_id} for '{defect_type}' (conf={confidence:.3f})")

            except Exception as e:
                logger.warning(f"[Tracker] Failed to initialize '{defect_type}': {e}")
                continue

        self._current_frame_idx = frame_idx
        logger.info(f"[Tracker] Initialized {len(new_tracks)} tracks on frame {frame_idx}")

        return new_tracks

    def propagate_tracks(
        self,
        frame: np.ndarray,
        frame_idx: int,
    ) -> Dict[int, np.ndarray]:
        """
        Propagate existing tracks to next frame.

        Args:
            frame: RGB image (HWC)
            frame_idx: Frame index

        Returns:
            Dict mapping track_id to mask for this frame
        """
        if not self.is_loaded():
            raise RuntimeError("Tracker not loaded. Call load() first.")

        import torch
        from PIL import Image

        logger.info(f"[Tracker] Propagating {len([t for t in self._tracks.values() if t.is_active])} active tracks to frame {frame_idx}")

        # Convert to PIL
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
        frame_pil = Image.fromarray(frame)

        frame_masks = {}

        # Propagate each active track
        for track_id, track in self._tracks.items():
            if not track.is_active:
                continue

            try:
                # Get previous mask as prompt
                prev_mask = track.masks[-1]

                # Convert mask to point prompts (sample points from mask)
                points = self._sample_points_from_mask(prev_mask, num_points=5)

                if len(points) == 0:
                    logger.warning(f"[Tracker] Track {track_id} has no valid points, deactivating")
                    track.is_active = False
                    continue

                # Prepare inputs with point prompts
                inputs = self._processor(
                    images=frame_pil,
                    input_points=[[points]],  # Batch of 1 image, 1 set of points
                    return_tensors="pt"
                ).to(self._device)

                # Run model
                with torch.no_grad():
                    outputs = self._model(**inputs)

                # Post-process
                results = self._processor.post_process_instance_segmentation(
                    outputs,
                    threshold=self.config.score_threshold,
                    mask_threshold=self.config.mask_threshold,
                    target_sizes=[[frame.shape[0], frame.shape[1]]]
                )[0]

                # Get propagated mask
                if len(results.get('masks', [])) > 0:
                    mask_tensor = results['masks'][0]
                    if hasattr(mask_tensor, 'cpu'):
                        mask_data = mask_tensor.cpu().numpy()
                    else:
                        mask_data = np.array(mask_tensor)

                    if mask_data.ndim > 2:
                        mask_data = mask_data.squeeze()
                    mask_binary = (mask_data > 0).astype(np.uint8)

                    # Get confidence
                    confidence = 1.0
                    if len(results.get('scores', [])) > 0:
                        score_tensor = results['scores'][0]
                        if hasattr(score_tensor, 'cpu'):
                            confidence = float(score_tensor.cpu().numpy())
                        else:
                            confidence = float(score_tensor)

                    # Update track
                    track.masks.append(mask_binary)
                    track.confidences.append(confidence)
                    track.frame_indices.append(frame_idx)

                    frame_masks[track_id] = mask_binary

                    logger.debug(f"[Tracker] Propagated track {track_id} to frame {frame_idx} (conf={confidence:.3f})")

                else:
                    # Lost tracking
                    logger.warning(f"[Tracker] Lost track {track_id} at frame {frame_idx}")
                    track.is_active = False

            except Exception as e:
                logger.error(f"[Tracker] Error propagating track {track_id}: {e}")
                track.is_active = False
                continue

        self._current_frame_idx = frame_idx
        logger.info(f"[Tracker] Propagated {len(frame_masks)}/{len(self._tracks)} tracks to frame {frame_idx}")

        return frame_masks

    def add_point_prompt(
        self,
        frame: np.ndarray,
        frame_idx: int,
        point_xy: Tuple[int, int],
        defect_type: str,
        is_positive: bool = True,
    ) -> Optional[TrackingState]:
        """
        Add a new object to track via point prompt on a specific frame.

        Args:
            frame: RGB image (HWC)
            frame_idx: Frame index
            point_xy: Point coordinates (x, y)
            defect_type: Type of defect
            is_positive: True for positive point (foreground), False for negative

        Returns:
            New tracking state if successful
        """
        if not self.is_loaded():
            raise RuntimeError("Tracker not loaded. Call load() first.")

        import torch
        from PIL import Image

        logger.info(f"[Tracker] Adding point prompt at {point_xy} on frame {frame_idx} for '{defect_type}'")

        # Convert to PIL
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
        frame_pil = Image.fromarray(frame)

        try:
            # Prepare inputs with point prompt
            inputs = self._processor(
                images=frame_pil,
                input_points=[[[point_xy]]],  # Batch of 1 image, 1 point
                input_labels=[[1 if is_positive else 0]],
                return_tensors="pt"
            ).to(self._device)

            # Run model
            with torch.no_grad():
                outputs = self._model(**inputs)

            # Post-process
            results = self._processor.post_process_instance_segmentation(
                outputs,
                threshold=self.config.score_threshold,
                mask_threshold=self.config.mask_threshold,
                target_sizes=[[frame.shape[0], frame.shape[1]]]
            )[0]

            if len(results.get('masks', [])) > 0:
                mask_tensor = results['masks'][0]
                if hasattr(mask_tensor, 'cpu'):
                    mask_data = mask_tensor.cpu().numpy()
                else:
                    mask_data = np.array(mask_tensor)

                if mask_data.ndim > 2:
                    mask_data = mask_data.squeeze()
                mask_binary = (mask_data > 0).astype(np.uint8)

                # Get confidence
                confidence = 1.0
                if len(results.get('scores', [])) > 0:
                    score_tensor = results['scores'][0]
                    if hasattr(score_tensor, 'cpu'):
                        confidence = float(score_tensor.cpu().numpy())
                    else:
                        confidence = float(score_tensor)

                # Create new track
                track = TrackingState(
                    track_id=self._next_track_id,
                    defect_type=defect_type,
                    masks=[mask_binary],
                    confidences=[confidence],
                    frame_indices=[frame_idx],
                    is_active=True,
                )

                self._tracks[self._next_track_id] = track
                self._next_track_id += 1

                logger.info(f"[Tracker] Created track {track.track_id} from point prompt (conf={confidence:.3f})")

                return track

        except Exception as e:
            logger.error(f"[Tracker] Failed to add point prompt: {e}")
            return None

    def _sample_points_from_mask(
        self,
        mask: np.ndarray,
        num_points: int = 5
    ) -> List[Tuple[int, int]]:
        """Sample representative points from a binary mask."""
        y_coords, x_coords = np.where(mask > 0)

        if len(y_coords) == 0:
            return []

        # Sample evenly spaced indices
        if len(y_coords) <= num_points:
            indices = range(len(y_coords))
        else:
            indices = np.linspace(0, len(y_coords) - 1, num_points, dtype=int)

        points = [(int(x_coords[i]), int(y_coords[i])) for i in indices]
        return points

    def get_active_tracks(self) -> List[TrackingState]:
        """Get all currently active tracks."""
        return [t for t in self._tracks.values() if t.is_active]

    def get_track(self, track_id: int) -> Optional[TrackingState]:
        """Get a specific track by ID."""
        return self._tracks.get(track_id)

    def deactivate_track(self, track_id: int) -> None:
        """Deactivate a specific track."""
        if track_id in self._tracks:
            self._tracks[track_id].is_active = False
            logger.info(f"[Tracker] Deactivated track {track_id}")
