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

            # Import SAM3 modules (Transformers implementation)
            from transformers import Sam3Model, Sam3Processor
            import torch

            # Load model from HuggingFace
            if self.config.model_path and self.config.model_path.exists():
                # Load local checkpoint
                self._model = Sam3Model.from_pretrained(
                    str(self.config.model_path.parent),
                    local_files_only=True
                ).to(self._device)
                self._processor = Sam3Processor.from_pretrained(
                    str(self.config.model_path.parent),
                    local_files_only=True
                )
            else:
                # Load from HuggingFace Hub
                self._model = Sam3Model.from_pretrained("facebook/sam3").to(self._device)
                self._processor = Sam3Processor.from_pretrained("facebook/sam3")

            self._predictor = self._processor  # For compatibility
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

        Args:
            image: RGB image (H, W, 3)
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load() first.")

        from PIL import Image
        # Convert numpy to PIL Image
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        self._current_image = Image.fromarray(image)

    def clear_image(self) -> None:
        """Clear the current image."""
        self._current_image = None

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

            # Use defect type as text prompt for SAM3
            text_prompt = detection.class_name if hasattr(detection, 'class_name') else "anomaly"

            # Prepare inputs for SAM3
            inputs = self._processor(
                images=self._current_image,
                text=text_prompt,
                return_tensors="pt"
            ).to(self._device)

            # Run prediction
            with torch.no_grad():
                outputs = self._model(**inputs)

            # Post-process results
            results = self._processor.post_process_instance_segmentation(
                outputs,
                threshold=self.config.mask_threshold,
                mask_threshold=self.config.mask_threshold,
                target_sizes=inputs.get("original_sizes").tolist()
            )[0]

            if len(results['masks']) == 0:
                return SegmentationResult(
                    detection_id=detection.detection_id,
                    mask=None,
                    success=False,
                    error_message="No mask generated",
                )

            # Get the first/best mask
            mask_data = results['masks'][0].cpu().numpy()
            sam_score = float(results['scores'][0].cpu().numpy()) if len(results['scores']) > 0 else 1.0

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
            import torch

            # SAM3 uses text prompts, use generic "object" for point-based segmentation
            text_prompt = "object"

            # Prepare inputs
            inputs = self._processor(
                images=self._current_image,
                text=text_prompt,
                return_tensors="pt"
            ).to(self._device)

            # Run prediction
            with torch.no_grad():
                outputs = self._model(**inputs)

            # Post-process results
            results = self._processor.post_process_instance_segmentation(
                outputs,
                threshold=self.config.mask_threshold,
                mask_threshold=self.config.mask_threshold,
                target_sizes=inputs.get("original_sizes").tolist()
            )[0]

            if len(results['masks']) == 0:
                return SegmentationResult(
                    detection_id="point_segment",
                    mask=None,
                    success=False,
                    error_message="No mask generated",
                )

            # Get the first mask
            mask_data = results['masks'][0].cpu().numpy()
            sam_score = float(results['scores'][0].cpu().numpy()) if len(results['scores']) > 0 else 1.0

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
