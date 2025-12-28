"""
SAM3 (Segment Anything Model 3) based segmentation refinement.

Uses Meta's official SAM3 implementation for pixel-accurate segmentation.
This module takes bounding box detections and generates precise masks
using SAM3's box-prompted or concept-prompted segmentation capability.

Key features of SAM3:
    - Unified 848M parameter model combining detector and tracker
    - Supports text/concept prompts (e.g., "crack", "corrosion")
    - Supports visual prompts (bounding boxes, points)
    - Works on both images and videos
    - Doubles accuracy over previous SAM versions

License: SAM License (commercial use permitted with restrictions)
    - No military, nuclear, espionage, or sanctioned entity use
    - See: https://github.com/facebookresearch/sam3/blob/main/LICENSE

Source: https://github.com/facebookresearch/sam3
Weights: https://huggingface.co/facebook/sam3
"""

from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import numpy as np

from anomaly_detection.models.data_models import (
    Detection,
    SegmentationMask,
)


@dataclass
class SegmenterConfig:
    """
    Configuration for SAM3 segmenter.

    Attributes:
        model_path: Path to SAM3 checkpoint (sam3.pt)
        device: Compute device ('cuda', 'cpu', or 'auto')
        mask_threshold: Threshold for binary mask conversion
        use_text_prompts: Use class names as text prompts for better accuracy
        max_masks_per_detection: Maximum masks to generate per detection
        hf_token: HuggingFace token for model download (optional)
    """
    model_path: Path
    device: str = "auto"
    mask_threshold: float = 0.0  # SAM outputs logits, 0.0 is standard threshold
    use_text_prompts: bool = True  # SAM3 supports text prompts
    max_masks_per_detection: int = 1  # Usually want single best mask
    hf_token: Optional[str] = None  # For downloading from HuggingFace

    def __post_init__(self):
        if self.max_masks_per_detection < 1:
            raise ValueError("max_masks_per_detection must be >= 1")


@dataclass
class SegmentationResult:
    """Result of segmenting a single detection."""
    detection_id: str
    mask: SegmentationMask
    success: bool
    error_message: Optional[str] = None


@dataclass
class SegmenterOutput:
    """Output container for segmenter results."""
    results: list[SegmentationResult]
    successful_count: int
    failed_count: int
    inference_time_ms: float


class SAM3Segmenter:
    """
    SAM3-based segmenter for refining detection bounding boxes to pixel masks.

    Uses the latest SAM3 model from Meta which supports:
    - Box prompts (primary mode for Phase-1)
    - Text/concept prompts (optional, can improve accuracy)
    - Point prompts
    - Mask prompts

    Usage:
        config = SegmenterConfig(model_path=Path("weights/sam3.pt"))
        segmenter = SAM3Segmenter(config)
        segmenter.load()

        # Segment a single detection
        result = segmenter.segment_detection(image, detection)

        # Or batch process
        output = segmenter.segment_batch(image, detections)

    Note: SAM3 weights available at:
        https://huggingface.co/facebook/sam3
    """

    def __init__(self, config: SegmenterConfig):
        """
        Initialize the segmenter.

        Args:
            config: Segmenter configuration
        """
        self.config = config
        self._model = None
        self._processor = None
        self._device = None
        self._current_image_shape: Optional[tuple[int, int]] = None
        self._inference_state = None  # SAM3 inference state

    def load(self) -> None:
        """
        Load SAM3 model into memory.

        Raises:
            RuntimeError: If SAM3 is not installed or loading fails
        """
        import torch

        # Determine device
        if self.config.device == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = self.config.device

        try:
            self._load_sam3()
        except ImportError as e:
            raise RuntimeError(
                "SAM3 not installed. Install with:\n"
                "  pip install git+https://github.com/facebookresearch/sam3.git\n"
                f"Original error: {e}"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load SAM3 model: {e}")

    def _load_sam3(self) -> None:
        """Load SAM3 model from official repository."""
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        if not self.config.model_path.exists():
            raise RuntimeError(
                f"SAM3 model file not found: {self.config.model_path}\n"
                "Download from: https://huggingface.co/facebook/sam3"
            )

        # Build model
        self._model = build_sam3_image_model(
            checkpoint=str(self.config.model_path),
            device=self._device,
        )
        self._processor = Sam3Processor(self._model)

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None

    def set_image(self, image: np.ndarray) -> None:
        """
        Precompute image embedding for efficient multi-prompt inference.

        Call this once per image before segmenting multiple detections.

        Args:
            image: Input image as numpy array (H, W, C) in RGB format
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load() first.")

        if image.ndim != 3:
            raise ValueError(f"Expected 3D image array, got shape {image.shape}")

        from PIL import Image

        self._current_image_shape = (image.shape[0], image.shape[1])

        # Convert numpy array to PIL Image (SAM3 expects PIL Image)
        pil_image = Image.fromarray(image)

        # set_image returns the inference state
        self._inference_state = self._processor.set_image(pil_image)

    def clear_image(self) -> None:
        """Clear cached image embedding to free memory."""
        self._current_image_shape = None
        self._inference_state = None

    def segment_detection(
        self,
        detection: Detection,
        image: Optional[np.ndarray] = None,
    ) -> SegmentationResult:
        """
        Generate segmentation mask for a single detection.

        Args:
            detection: Detection with bounding box to segment
            image: Optional image if set_image() wasn't called

        Returns:
            SegmentationResult with mask or error
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load() first.")

        # Set image if provided
        if image is not None:
            self.set_image(image)
        elif self._current_image_shape is None:
            raise ValueError("No image set. Call set_image() or provide image argument.")

        try:
            mask, score = self._predict_mask(detection)

            seg_mask = SegmentationMask(
                mask=mask,
                sam_score=score,
            )

            return SegmentationResult(
                detection_id=detection.detection_id,
                mask=seg_mask,
                success=True,
            )

        except Exception as e:
            return SegmentationResult(
                detection_id=detection.detection_id,
                mask=SegmentationMask(
                    mask=np.zeros(self._current_image_shape, dtype=np.uint8),
                    sam_score=0.0,
                ),
                success=False,
                error_message=str(e),
            )

    def segment_batch(
        self,
        image: np.ndarray,
        detections: list[Detection],
    ) -> SegmenterOutput:
        """
        Segment multiple detections efficiently.

        Computes image embedding once and reuses for all detections.

        Args:
            image: Input image (H, W, C) in RGB format
            detections: List of detections to segment

        Returns:
            SegmenterOutput with all results
        """
        import time

        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load() first.")

        start_time = time.perf_counter()

        # Set image once
        self.set_image(image)

        results = []
        successful = 0
        failed = 0

        for detection in detections:
            result = self.segment_detection(detection)
            results.append(result)

            if result.success:
                successful += 1
            else:
                failed += 1

        # Clear embedding after batch
        self.clear_image()

        inference_time_ms = (time.perf_counter() - start_time) * 1000

        return SegmenterOutput(
            results=results,
            successful_count=successful,
            failed_count=failed,
            inference_time_ms=inference_time_ms,
        )

    def _predict_mask(self, detection: Detection) -> tuple[np.ndarray, float]:
        """
        Run SAM3 prediction with box and optional text prompt.

        Args:
            detection: Detection with bbox and class_name

        Returns:
            Tuple of (binary mask, confidence score)
        """
        if self._inference_state is None:
            raise RuntimeError("No image set. Call set_image() first.")

        # Use text prompt if enabled (SAM3 concept prompting feature)
        if self.config.use_text_prompts:
            # Use class name as concept prompt
            text_prompt = detection.class_name.replace("_", " ")
            output = self._processor.set_text_prompt(
                state=self._inference_state,
                prompt=text_prompt,
            )
        else:
            # Box prompt: [x_min, y_min, x_max, y_max]
            box = [
                detection.bbox.x_min,
                detection.bbox.y_min,
                detection.bbox.x_max,
                detection.bbox.y_max,
            ]
            output = self._processor.set_box_prompt(
                state=self._inference_state,
                box=box,
            )

        # Extract results from output dict
        masks = output.get("masks", [])
        scores = output.get("scores", [])

        if len(masks) == 0:
            # Return empty mask if no predictions
            empty_mask = np.zeros(self._current_image_shape, dtype=np.uint8)
            return empty_mask, 0.0

        # Select best mask
        if len(masks) > 1 and len(scores) > 1:
            best_idx = np.argmax(scores)
            mask = masks[best_idx]
            score = float(scores[best_idx])
        else:
            mask = masks[0]
            score = float(scores[0]) if len(scores) > 0 else 0.0

        # Convert to numpy if needed and ensure correct shape
        if not isinstance(mask, np.ndarray):
            import torch
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy()

        # Convert to binary uint8
        binary_mask = (mask > self.config.mask_threshold).astype(np.uint8)

        # Ensure 2D mask
        if binary_mask.ndim == 3:
            binary_mask = binary_mask.squeeze()

        return binary_mask, score

    def unload(self) -> None:
        """Release model from memory."""
        self.clear_image()
        self._model = None
        self._processor = None
        self._device = None

        import gc
        gc.collect()

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
