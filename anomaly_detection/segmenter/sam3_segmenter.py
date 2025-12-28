"""SAM3 segmenter for pixel-accurate mask generation."""

from dataclasses import dataclass
from pathlib import Path
import time
import numpy as np

from ..models import Detection, SegmentationMask


@dataclass
class SegmenterConfig:
    """Configuration for SAM3 segmenter."""
    model_path: Path | None = None
    device: str = "auto"
    mask_threshold: float = 0.0
    use_text_prompts: bool = False
    max_masks_per_detection: int = 1
    hf_token: str | None = None


@dataclass
class SegmentationResult:
    """Result from segmentation."""
    detection_id: str
    mask: SegmentationMask | None
    success: bool
    error_message: str | None = None


class SAM3Segmenter:
    """SAM3 segmenter for generating pixel-accurate masks."""

    def __init__(self, config: SegmenterConfig | None = None):
        self.config = config or SegmenterConfig()
        self._model = None
        self._predictor = None
        self._device = None
        self._is_loaded = False
        self._current_image = None
        self._inference_state = None

    def load(self) -> None:
        """Load the SAM3 model."""
        if self._is_loaded:
            return

        try:
            import torch

            # Determine device
            if self.config.device == "auto":
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self._device = self.config.device

            # Import SAM3 modules
            from sam3.build_sam import build_sam3
            from sam3.sam3_image_predictor import SAM3ImagePredictor

            # Load model
            if self.config.model_path and self.config.model_path.exists():
                checkpoint = str(self.config.model_path)
                checkpoint_lower = checkpoint.lower()

                # Determine config from checkpoint name
                if "large" in checkpoint_lower:
                    model_cfg = "sam3_hiera_l.yaml"
                elif "base" in checkpoint_lower:
                    model_cfg = "sam3_hiera_b+.yaml"
                elif "small" in checkpoint_lower:
                    model_cfg = "sam3_hiera_s.yaml"
                elif "tiny" in checkpoint_lower:
                    model_cfg = "sam3_hiera_t.yaml"
                else:
                    # Default: use base config for generic sam3.pt file
                    model_cfg = "sam3_hiera_l.yaml"

                self._model = build_sam3(model_cfg, checkpoint, device=self._device)
            else:
                # Try to load from HuggingFace
                from sam3.build_sam import build_sam3_hf
                self._model = build_sam3_hf(
                    "facebook/sam3-hiera-large",
                    device=self._device,
                )

            self._predictor = SAM3ImagePredictor(self._model)
            self._is_loaded = True

        except ImportError as e:
            raise ImportError(
                f"SAM3 not installed. Install: pip install git+https://github.com/facebookresearch/sam3.git\nError: {e}"
            )

    def unload(self) -> None:
        """Unload model from memory."""
        self.clear_image()

        if self._predictor is not None:
            del self._predictor
            self._predictor = None

        if self._model is not None:
            del self._model
            self._model = None

        self._is_loaded = False

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def is_loaded(self) -> bool:
        return self._is_loaded and self._predictor is not None

    def set_image(self, image: np.ndarray) -> None:
        """
        Set the current image for segmentation.

        This pre-computes image embeddings for efficient multi-detection processing.

        Args:
            image: RGB image (H, W, 3)
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load() first.")

        self._current_image = image
        self._predictor.set_image(image)

    def clear_image(self) -> None:
        """Clear the current image and embeddings."""
        self._current_image = None
        if self._predictor is not None:
            self._predictor.reset_predictor()

    def segment_detection(
        self,
        detection: Detection,
        image: np.ndarray | None = None,
    ) -> SegmentationResult:
        """
        Segment a single detection.

        Args:
            detection: Detection to segment
            image: Optional image (uses cached if not provided)

        Returns:
            SegmentationResult with mask
        """
        if not self.is_loaded():
            return SegmentationResult(
                detection_id=detection.detection_id,
                mask=None,
                success=False,
                error_message="Model not loaded",
            )

        try:
            # Set image if provided
            if image is not None:
                self.set_image(image)
            elif self._current_image is None:
                return SegmentationResult(
                    detection_id=detection.detection_id,
                    mask=None,
                    success=False,
                    error_message="No image set",
                )

            import torch

            # Prepare box prompt
            bbox = detection.bbox
            input_box = np.array([bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max])

            # Run prediction
            masks, scores, _ = self._predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box,
                multimask_output=True,
            )

            # Select best mask
            best_idx = np.argmax(scores)
            mask_data = masks[best_idx]
            sam_score = float(scores[best_idx])

            # Convert to binary
            mask_binary = (mask_data > self.config.mask_threshold).astype(np.uint8)

            return SegmentationResult(
                detection_id=detection.detection_id,
                mask=SegmentationMask(
                    data=mask_binary,
                    sam_score=sam_score,
                ),
                success=True,
            )

        except Exception as e:
            return SegmentationResult(
                detection_id=detection.detection_id,
                mask=None,
                success=False,
                error_message=str(e),
            )

    def segment_batch(
        self,
        image: np.ndarray,
        detections: list[Detection],
    ) -> list[SegmentationResult]:
        """
        Segment multiple detections efficiently.

        Args:
            image: RGB image
            detections: List of detections to segment

        Returns:
            List of SegmentationResults
        """
        if not detections:
            return []

        self.set_image(image)

        try:
            results = []
            for detection in detections:
                result = self.segment_detection(detection)
                results.append(result)
            return results
        finally:
            self.clear_image()

    def segment_with_points(
        self,
        image: np.ndarray,
        points: list[tuple[int, int]],
        labels: list[int] | None = None,
    ) -> SegmentationResult:
        """
        Segment using point prompts.

        Args:
            image: RGB image
            points: List of (x, y) points
            labels: List of labels (1=foreground, 0=background)

        Returns:
            SegmentationResult
        """
        if not self.is_loaded():
            return SegmentationResult(
                detection_id="point_segment",
                mask=None,
                success=False,
                error_message="Model not loaded",
            )

        try:
            self.set_image(image)

            # Prepare inputs
            input_points = np.array(points)
            input_labels = np.array(labels if labels else [1] * len(points))

            # Run prediction
            masks, scores, _ = self._predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                box=None,
                multimask_output=True,
            )

            # Select best mask
            best_idx = np.argmax(scores)
            mask_data = masks[best_idx]
            sam_score = float(scores[best_idx])

            mask_binary = (mask_data > self.config.mask_threshold).astype(np.uint8)

            return SegmentationResult(
                detection_id="point_segment",
                mask=SegmentationMask(
                    data=mask_binary,
                    sam_score=sam_score,
                ),
                success=True,
            )

        except Exception as e:
            return SegmentationResult(
                detection_id="point_segment",
                mask=None,
                success=False,
                error_message=str(e),
            )
        finally:
            self.clear_image()

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unload()
