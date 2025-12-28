"""SAM3 segmenter for pixel-accurate mask generation."""

from dataclasses import dataclass
from pathlib import Path
import time
import logging
import numpy as np

from ..models import Detection, SegmentationMask

logger = logging.getLogger(__name__)


@dataclass
class SegmenterConfig:
    """Configuration for SAM3 segmenter."""
    model_path: Path | None = None
    device: str = "auto"
    mask_threshold: float = 0.5
    score_threshold: float = 0.5
    use_text_prompts: bool = True
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
        self._processor = None
        self._device = None
        self._is_loaded = False
        self._current_image = None
        self._current_image_size = None

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

            logger.info(f"Loading SAM3 model on device: {self._device}")

            # Import SAM3 modules (Transformers implementation)
            from transformers import Sam3Model, Sam3Processor

            # Load model from HuggingFace Hub
            model_kwargs = {}
            if self.config.hf_token:
                model_kwargs["token"] = self.config.hf_token

            self._model = Sam3Model.from_pretrained("facebook/sam3", **model_kwargs).to(self._device)
            self._processor = Sam3Processor.from_pretrained("facebook/sam3", **model_kwargs)

            self._is_loaded = True
            logger.info("SAM3 model loaded successfully")

        except ImportError as e:
            raise ImportError(
                f"SAM3 not installed. Install transformers with SAM3 support.\nError: {e}"
            )

    def unload(self) -> None:
        """Unload model from memory."""
        self.clear_image()

        if self._processor is not None:
            del self._processor
            self._processor = None

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
        return self._is_loaded and self._processor is not None

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
        self._current_image_size = (image.shape[0], image.shape[1])  # (H, W)

    def clear_image(self) -> None:
        """Clear the current image."""
        self._current_image = None
        self._current_image_size = None

    def segment_detection(
        self,
        detection: Detection,
        image: np.ndarray | None = None,
    ) -> SegmentationResult:
        """
        Segment a single detection using text prompt.

        Args:
            detection: Detection to segment (uses class_name as text prompt)
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
            logger.debug(f"SAM3 segmenting with text prompt: '{text_prompt}'")

            # Prepare inputs for SAM3
            inputs = self._processor(
                images=self._current_image,
                text=text_prompt,
                return_tensors="pt"
            ).to(self._device)

            # Run prediction
            with torch.no_grad():
                outputs = self._model(**inputs)

            # Get original sizes from inputs for post-processing
            original_sizes = inputs.get("original_sizes")
            if original_sizes is not None:
                target_sizes = original_sizes.tolist()
            else:
                # Fallback to stored image size
                target_sizes = [list(self._current_image_size)] if self._current_image_size else None

            # Post-process results
            results = self._processor.post_process_instance_segmentation(
                outputs,
                threshold=self.config.score_threshold,
                mask_threshold=self.config.mask_threshold,
                target_sizes=target_sizes
            )[0]

            if len(results['masks']) == 0:
                logger.debug(f"SAM3: No mask generated for prompt '{text_prompt}'")
                return SegmentationResult(
                    detection_id=detection.detection_id,
                    mask=None,
                    success=False,
                    error_message=f"No mask generated for '{text_prompt}'",
                )

            # Get the first/best mask
            mask_tensor = results['masks'][0]
            if hasattr(mask_tensor, 'cpu'):
                mask_data = mask_tensor.cpu().numpy()
            else:
                mask_data = np.array(mask_tensor)

            # Get score
            if len(results.get('scores', [])) > 0:
                score_tensor = results['scores'][0]
                if hasattr(score_tensor, 'cpu'):
                    sam_score = float(score_tensor.cpu().numpy())
                else:
                    sam_score = float(score_tensor)
            else:
                sam_score = 1.0

            # Ensure mask is 2D binary
            if mask_data.ndim > 2:
                mask_data = mask_data.squeeze()
            mask_binary = (mask_data > 0).astype(np.uint8)

            # Check if mask has any content
            mask_area = mask_binary.sum()
            if mask_area == 0:
                logger.debug(f"SAM3: Empty mask for prompt '{text_prompt}'")
                return SegmentationResult(
                    detection_id=detection.detection_id,
                    mask=None,
                    success=False,
                    error_message=f"Empty mask for '{text_prompt}'",
                )

            logger.debug(f"SAM3: Found mask for '{text_prompt}', score={sam_score:.3f}, area={mask_area}")

            return SegmentationResult(
                detection_id=detection.detection_id,
                mask=SegmentationMask(
                    data=mask_binary,
                    sam_score=sam_score,
                ),
                success=True,
            )

        except Exception as e:
            logger.error(f"SAM3 segmentation error: {e}")
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

    def segment_with_box(
        self,
        image: np.ndarray,
        box_xyxy: list[int],
        detection_id: str = "box_segment",
    ) -> SegmentationResult:
        """
        Segment using a bounding box prompt.

        Args:
            image: RGB image
            box_xyxy: Bounding box as [x1, y1, x2, y2]
            detection_id: Optional ID for the result

        Returns:
            SegmentationResult
        """
        if not self.is_loaded():
            return SegmentationResult(
                detection_id=detection_id,
                mask=None,
                success=False,
                error_message="Model not loaded",
            )

        try:
            self.set_image(image)
            import torch

            # Prepare inputs with box prompt
            inputs = self._processor(
                images=self._current_image,
                text=None,
                input_boxes=[[box_xyxy]],
                input_boxes_labels=[[1]],  # Positive box
                return_tensors="pt"
            ).to(self._device)

            # Run prediction
            with torch.no_grad():
                outputs = self._model(**inputs)

            # Get original sizes
            original_sizes = inputs.get("original_sizes")
            if original_sizes is not None:
                target_sizes = original_sizes.tolist()
            else:
                target_sizes = [list(self._current_image_size)] if self._current_image_size else None

            # Post-process results
            results = self._processor.post_process_instance_segmentation(
                outputs,
                threshold=self.config.score_threshold,
                mask_threshold=self.config.mask_threshold,
                target_sizes=target_sizes
            )[0]

            if len(results['masks']) == 0:
                return SegmentationResult(
                    detection_id=detection_id,
                    mask=None,
                    success=False,
                    error_message="No mask generated from box",
                )

            # Get the first mask
            mask_tensor = results['masks'][0]
            if hasattr(mask_tensor, 'cpu'):
                mask_data = mask_tensor.cpu().numpy()
            else:
                mask_data = np.array(mask_tensor)

            # Get score
            if len(results.get('scores', [])) > 0:
                score_tensor = results['scores'][0]
                if hasattr(score_tensor, 'cpu'):
                    sam_score = float(score_tensor.cpu().numpy())
                else:
                    sam_score = float(score_tensor)
            else:
                sam_score = 1.0

            # Ensure mask is 2D binary
            if mask_data.ndim > 2:
                mask_data = mask_data.squeeze()
            mask_binary = (mask_data > 0).astype(np.uint8)

            return SegmentationResult(
                detection_id=detection_id,
                mask=SegmentationMask(
                    data=mask_binary,
                    sam_score=sam_score,
                ),
                success=True,
            )

        except Exception as e:
            logger.error(f"SAM3 box segmentation error: {e}")
            return SegmentationResult(
                detection_id=detection_id,
                mask=None,
                success=False,
                error_message=str(e),
            )
        finally:
            self.clear_image()

    def segment_with_point(
        self,
        image: np.ndarray,
        point_xy: tuple[int, int],
        detection_id: str = "point_segment",
        label: int = 1,
    ) -> SegmentationResult:
        """
        Segment using a point prompt.

        Args:
            image: RGB image
            point_xy: Point as (x, y) coordinates
            detection_id: Optional ID for the result
            label: 1 for positive (foreground), 0 for negative (background)

        Returns:
            SegmentationResult
        """
        if not self.is_loaded():
            return SegmentationResult(
                detection_id=detection_id,
                mask=None,
                success=False,
                error_message="Model not loaded",
            )

        try:
            self.set_image(image)
            import torch

            # Prepare inputs with point prompt
            inputs = self._processor(
                images=self._current_image,
                text=None,
                input_points=[[[point_xy[0], point_xy[1]]]],
                input_labels=[[label]],
                return_tensors="pt"
            ).to(self._device)

            # Run prediction
            with torch.no_grad():
                outputs = self._model(**inputs)

            # Get original sizes
            original_sizes = inputs.get("original_sizes")
            if original_sizes is not None:
                target_sizes = original_sizes.tolist()
            else:
                target_sizes = [list(self._current_image_size)] if self._current_image_size else None

            # Post-process results
            results = self._processor.post_process_instance_segmentation(
                outputs,
                threshold=self.config.score_threshold,
                mask_threshold=self.config.mask_threshold,
                target_sizes=target_sizes
            )[0]

            if len(results['masks']) == 0:
                return SegmentationResult(
                    detection_id=detection_id,
                    mask=None,
                    success=False,
                    error_message="No mask generated from point",
                )

            # Get the first mask
            mask_tensor = results['masks'][0]
            if hasattr(mask_tensor, 'cpu'):
                mask_data = mask_tensor.cpu().numpy()
            else:
                mask_data = np.array(mask_tensor)

            # Get score
            if len(results.get('scores', [])) > 0:
                score_tensor = results['scores'][0]
                if hasattr(score_tensor, 'cpu'):
                    sam_score = float(score_tensor.cpu().numpy())
                else:
                    sam_score = float(score_tensor)
            else:
                sam_score = 1.0

            # Ensure mask is 2D binary
            if mask_data.ndim > 2:
                mask_data = mask_data.squeeze()
            mask_binary = (mask_data > 0).astype(np.uint8)

            logger.debug(f"SAM3 point segment: point={point_xy}, score={sam_score:.3f}, area={mask_binary.sum()}")

            return SegmentationResult(
                detection_id=detection_id,
                mask=SegmentationMask(
                    data=mask_binary,
                    sam_score=sam_score,
                ),
                success=True,
            )

        except Exception as e:
            logger.error(f"SAM3 point segmentation error: {e}")
            return SegmentationResult(
                detection_id=detection_id,
                mask=None,
                success=False,
                error_message=str(e),
            )
        finally:
            self.clear_image()

    def segment_with_text_batch(
        self,
        image: np.ndarray,
        text_prompts: list[str],
        prefix: str = "batch",
    ) -> list[SegmentationResult]:
        """
        Segment image using multiple text prompts in a TRUE batch (single forward pass).

        Args:
            image: RGB image (original, no grid overlay)
            text_prompts: List of defect type names to search for
            prefix: Prefix for detection IDs

        Returns:
            List of SegmentationResults (one per text prompt that found something)
        """
        if not self.is_loaded():
            return []

        results = []
        self.set_image(image)

        try:
            import torch

            # Try true batch processing first (all prompts in single forward pass)
            try:
                # Prepare batched inputs - replicate image for each text prompt
                images = [self._current_image] * len(text_prompts)

                inputs = self._processor(
                    images=images,
                    text=text_prompts,
                    return_tensors="pt",
                ).to(self._device)

                # Single forward pass for ALL prompts
                with torch.no_grad():
                    outputs = self._model(**inputs)

                # Get target sizes for all images in batch
                original_sizes = inputs.get("original_sizes")
                if original_sizes is not None:
                    target_sizes = original_sizes.tolist()
                else:
                    target_sizes = [list(self._current_image_size)] * len(text_prompts)

                # Post-process all results at once
                all_results = self._processor.post_process_instance_segmentation(
                    outputs,
                    threshold=self.config.score_threshold,
                    mask_threshold=self.config.mask_threshold,
                    target_sizes=target_sizes
                )

                # Extract results for each prompt
                for i, (text_prompt, proc_results) in enumerate(zip(text_prompts, all_results)):
                    detection_id = f"{prefix}_{text_prompt}_{i}"

                    if len(proc_results.get('masks', [])) == 0:
                        logger.debug(f"SAM3 batch: No mask for '{text_prompt}'")
                        continue

                    mask_tensor = proc_results['masks'][0]
                    if hasattr(mask_tensor, 'cpu'):
                        mask_data = mask_tensor.cpu().numpy()
                    else:
                        mask_data = np.array(mask_tensor)

                    # Get score
                    if len(proc_results.get('scores', [])) > 0:
                        score_tensor = proc_results['scores'][0]
                        if hasattr(score_tensor, 'cpu'):
                            sam_score = float(score_tensor.cpu().numpy())
                        else:
                            sam_score = float(score_tensor)
                    else:
                        sam_score = 1.0

                    # Ensure mask is 2D binary
                    if mask_data.ndim > 2:
                        mask_data = mask_data.squeeze()
                    mask_binary = (mask_data > 0).astype(np.uint8)

                    if mask_binary.sum() == 0:
                        continue

                    logger.debug(f"SAM3 batch: Found '{text_prompt}', score={sam_score:.3f}")

                    results.append(SegmentationResult(
                        detection_id=detection_id,
                        mask=SegmentationMask(
                            data=mask_binary,
                            sam_score=sam_score,
                        ),
                        success=True,
                    ))

                logger.info(f"SAM3 batch: Processed {len(text_prompts)} prompts in single forward pass, got {len(results)} masks")

            except Exception as batch_error:
                # Fallback to sequential if true batch fails
                logger.warning(f"SAM3 true batch failed ({batch_error}), falling back to sequential")
                results = self._segment_text_sequential(text_prompts, prefix)

        finally:
            self.clear_image()

        return results

    def _segment_text_sequential(
        self,
        text_prompts: list[str],
        prefix: str,
    ) -> list[SegmentationResult]:
        """Fallback sequential text prompt processing."""
        import torch
        results = []

        for i, text_prompt in enumerate(text_prompts):
            detection_id = f"{prefix}_{text_prompt}_{i}"

            try:
                inputs = self._processor(
                    images=self._current_image,
                    text=text_prompt,
                    return_tensors="pt"
                ).to(self._device)

                with torch.no_grad():
                    outputs = self._model(**inputs)

                original_sizes = inputs.get("original_sizes")
                if original_sizes is not None:
                    target_sizes = original_sizes.tolist()
                else:
                    target_sizes = [list(self._current_image_size)] if self._current_image_size else None

                proc_results = self._processor.post_process_instance_segmentation(
                    outputs,
                    threshold=self.config.score_threshold,
                    mask_threshold=self.config.mask_threshold,
                    target_sizes=target_sizes
                )[0]

                if len(proc_results['masks']) == 0:
                    continue

                mask_tensor = proc_results['masks'][0]
                if hasattr(mask_tensor, 'cpu'):
                    mask_data = mask_tensor.cpu().numpy()
                else:
                    mask_data = np.array(mask_tensor)

                if len(proc_results.get('scores', [])) > 0:
                    score_tensor = proc_results['scores'][0]
                    if hasattr(score_tensor, 'cpu'):
                        sam_score = float(score_tensor.cpu().numpy())
                    else:
                        sam_score = float(score_tensor)
                else:
                    sam_score = 1.0

                if mask_data.ndim > 2:
                    mask_data = mask_data.squeeze()
                mask_binary = (mask_data > 0).astype(np.uint8)

                if mask_binary.sum() == 0:
                    continue

                results.append(SegmentationResult(
                    detection_id=detection_id,
                    mask=SegmentationMask(
                        data=mask_binary,
                        sam_score=sam_score,
                    ),
                    success=True,
                ))

            except Exception as e:
                logger.warning(f"SAM3 sequential error for '{text_prompt}': {e}")
                continue

        return results

    def segment_tiles_with_text(
        self,
        image: np.ndarray,
        text_prompts: list[str],
        grid_cols: int = 3,
        grid_rows: int = 3,
        prefix: str = "tile",
    ) -> list[SegmentationResult]:
        """
        Segment image tiles using text prompts - BATCHED across all tiles.

        Batches all (tile, prompt) pairs into a single forward pass for efficiency.

        Args:
            image: RGB image (original, no grid overlay)
            text_prompts: List of defect type names to search for
            grid_cols: Number of columns in grid
            grid_rows: Number of rows in grid
            prefix: Prefix for detection IDs

        Returns:
            List of SegmentationResults with masks in full image coordinates
        """
        if not self.is_loaded():
            return []

        from PIL import Image
        import torch

        h, w = image.shape[:2]
        tile_h = h // grid_rows
        tile_w = w // grid_cols

        # Collect all tiles and their metadata
        tiles_data = []  # [(tile_pil, tile_label, y_start, x_start, y_end, x_end), ...]

        for row in range(grid_rows):
            for col in range(grid_cols):
                y_start = row * tile_h
                x_start = col * tile_w
                y_end = h if row == grid_rows - 1 else (row + 1) * tile_h
                x_end = w if col == grid_cols - 1 else (col + 1) * tile_w

                tile = image[y_start:y_end, x_start:x_end]
                tile_label = f"{chr(65 + row)}{col + 1}"

                # Convert to PIL
                if tile.dtype != np.uint8:
                    tile = (tile * 255).astype(np.uint8)
                tile_pil = Image.fromarray(tile)

                tiles_data.append((tile_pil, tile_label, y_start, x_start, y_end, x_end))

        all_results = []

        try:
            # Build mega-batch: all tiles × all prompts
            all_images = []
            all_texts = []
            all_metadata = []  # (tile_idx, prompt_idx, tile_label, bounds)

            for tile_idx, (tile_pil, tile_label, y_start, x_start, y_end, x_end) in enumerate(tiles_data):
                for prompt_idx, text_prompt in enumerate(text_prompts):
                    all_images.append(tile_pil)
                    all_texts.append(text_prompt)
                    all_metadata.append({
                        'tile_idx': tile_idx,
                        'prompt_idx': prompt_idx,
                        'tile_label': tile_label,
                        'text_prompt': text_prompt,
                        'y_start': y_start,
                        'x_start': x_start,
                        'tile_h': y_end - y_start,
                        'tile_w': x_end - x_start,
                    })

            logger.info(f"SAM3 tiles: Processing {len(all_images)} (tile,prompt) pairs in batched mode")

            # Process in chunks to avoid OOM (e.g., 12 at a time)
            chunk_size = min(12, len(all_images))  # Process up to 12 at once

            for chunk_start in range(0, len(all_images), chunk_size):
                chunk_end = min(chunk_start + chunk_size, len(all_images))
                chunk_images = all_images[chunk_start:chunk_end]
                chunk_texts = all_texts[chunk_start:chunk_end]
                chunk_meta = all_metadata[chunk_start:chunk_end]

                try:
                    inputs = self._processor(
                        images=chunk_images,
                        text=chunk_texts,
                        return_tensors="pt",
                    ).to(self._device)

                    with torch.no_grad():
                        outputs = self._model(**inputs)

                    # Build target sizes for this chunk
                    target_sizes = [[m['tile_h'], m['tile_w']] for m in chunk_meta]

                    proc_results = self._processor.post_process_instance_segmentation(
                        outputs,
                        threshold=self.config.score_threshold,
                        mask_threshold=self.config.mask_threshold,
                        target_sizes=target_sizes
                    )

                    # Extract results
                    for meta, result in zip(chunk_meta, proc_results):
                        if len(result.get('masks', [])) == 0:
                            continue

                        mask_tensor = result['masks'][0]
                        if hasattr(mask_tensor, 'cpu'):
                            mask_data = mask_tensor.cpu().numpy()
                        else:
                            mask_data = np.array(mask_tensor)

                        if len(result.get('scores', [])) > 0:
                            score_tensor = result['scores'][0]
                            if hasattr(score_tensor, 'cpu'):
                                sam_score = float(score_tensor.cpu().numpy())
                            else:
                                sam_score = float(score_tensor)
                        else:
                            sam_score = 1.0

                        if mask_data.ndim > 2:
                            mask_data = mask_data.squeeze()
                        mask_binary = (mask_data > 0).astype(np.uint8)

                        if mask_binary.sum() == 0:
                            continue

                        # Transform to full image coordinates
                        full_mask = np.zeros((h, w), dtype=np.uint8)
                        full_mask[meta['y_start']:meta['y_start'] + mask_binary.shape[0],
                                  meta['x_start']:meta['x_start'] + mask_binary.shape[1]] = mask_binary

                        detection_id = f"{prefix}_{meta['tile_label']}_{meta['text_prompt']}_{meta['prompt_idx']}"

                        all_results.append(SegmentationResult(
                            detection_id=detection_id,
                            mask=SegmentationMask(
                                data=full_mask,
                                sam_score=sam_score,
                            ),
                            success=True,
                        ))

                except Exception as chunk_error:
                    logger.warning(f"SAM3 tile chunk error: {chunk_error}, processing individually")
                    # Fallback for this chunk
                    for img, txt, meta in zip(chunk_images, chunk_texts, chunk_meta):
                        try:
                            self._current_image = img
                            self._current_image_size = (meta['tile_h'], meta['tile_w'])
                            result = self._segment_text_sequential([txt], f"{prefix}_{meta['tile_label']}")
                            for r in result:
                                if r.success and r.mask:
                                    full_mask = np.zeros((h, w), dtype=np.uint8)
                                    full_mask[meta['y_start']:meta['y_start'] + r.mask.data.shape[0],
                                              meta['x_start']:meta['x_start'] + r.mask.data.shape[1]] = r.mask.data
                                    r.mask = SegmentationMask(data=full_mask, sam_score=r.mask.sam_score)
                                    all_results.append(r)
                        except Exception:
                            continue

        except Exception as e:
            logger.error(f"SAM3 tiles batch error: {e}")

        logger.info(f"SAM3 tiles: Found {len(all_results)} masks across {grid_cols * grid_rows} tiles")
        return all_results

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unload()
