"""
RunPod Serverless Handler.

This module provides the serverless handler function for RunPod.
It processes incoming requests and returns anomaly detection results.

Environment Variables:
    DETECTOR_VARIANT: RF-DETR variant (nano, small, medium, large)
    DETECTOR_WEIGHTS: Path to custom detector weights (optional)
    DETECTOR_NUM_CLASSES: Number of classes for custom model (optional)
    SAM_MODEL_PATH: Path to SAM3/SAM2 model weights
    CONFIDENCE_THRESHOLD: Detection confidence threshold (default: 0.3)
    SAVE_MASKS: Whether to save masks (default: false)
"""

import os
import base64
import json
import traceback
from pathlib import Path
from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image

# Import will happen after model loading
pipeline = None


def load_pipeline():
    """Load the pipeline on cold start."""
    global pipeline

    if pipeline is not None:
        return

    from anomaly_detection.pipeline import Phase1Pipeline, PipelineConfig, RFDETRVariant

    # Get configuration from environment
    variant_name = os.environ.get("DETECTOR_VARIANT", "medium").upper()
    variant = getattr(RFDETRVariant, variant_name, RFDETRVariant.MEDIUM)

    detector_weights = os.environ.get("DETECTOR_WEIGHTS")
    detector_num_classes = os.environ.get("DETECTOR_NUM_CLASSES")
    sam_model_path = os.environ.get("SAM_MODEL_PATH", "/models/sam3.pt")
    confidence = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.3"))
    save_masks = os.environ.get("SAVE_MASKS", "false").lower() == "true"

    config = PipelineConfig(
        segmenter_model_path=Path(sam_model_path),
        detector_variant=variant,
        detector_weights=Path(detector_weights) if detector_weights else None,
        detector_num_classes=int(detector_num_classes) if detector_num_classes else None,
        device="cuda",
        save_masks=save_masks,
        mask_output_dir=Path("/tmp/masks") if save_masks else None,
    )

    # Override confidence threshold
    if config.detector_config:
        config.detector_config.confidence_threshold = confidence

    pipeline = Phase1Pipeline(config)
    pipeline.load()
    print(f"Pipeline loaded: RF-DETR {variant.value} + SAM3")


def decode_image(image_data: str) -> np.ndarray:
    """Decode base64 image to numpy array."""
    # Handle data URL format
    if "," in image_data:
        image_data = image_data.split(",")[1]

    image_bytes = base64.b64decode(image_data)
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    return np.array(image)


def encode_mask(mask: np.ndarray) -> str:
    """Encode mask to base64 PNG."""
    img = Image.fromarray((mask * 255).astype(np.uint8))
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def handler(event: dict) -> dict:
    """
    RunPod serverless handler function.

    Args:
        event: Request event containing:
            - input.image: Base64 encoded image (required)
            - input.frame_id: Optional frame identifier
            - input.include_masks: Whether to include masks in response (default: false)
            - input.confidence_threshold: Override confidence threshold (optional)

    Returns:
        dict containing:
            - anomalies: List of detected anomalies with geometry
            - structures: List of detected structures
            - processing_time_ms: Total processing time
            - error: Error message if failed
    """
    try:
        # Load pipeline on first request
        load_pipeline()

        # Parse input
        input_data = event.get("input", {})

        if "image" not in input_data:
            return {"error": "Missing required field: image"}

        # Decode image
        image = decode_image(input_data["image"])

        # Get options
        frame_id = input_data.get("frame_id")
        include_masks = input_data.get("include_masks", False)

        # Override confidence if provided
        confidence = input_data.get("confidence_threshold")
        if confidence is not None and pipeline.config.detector_config:
            original_conf = pipeline.config.detector_config.confidence_threshold
            pipeline.config.detector_config.confidence_threshold = float(confidence)

        # Process image
        result = pipeline.process(image, frame_id=frame_id)

        # Restore confidence
        if confidence is not None and pipeline.config.detector_config:
            pipeline.config.detector_config.confidence_threshold = original_conf

        # Build response
        anomalies = []
        for anomaly in result.anomalies:
            anomaly_data = {
                "anomaly_id": anomaly.anomaly_id,
                "defect_type": anomaly.defect_type,
                "structure_type": anomaly.structure_type,
                "structure_id": anomaly.structure_id,
                "bbox": {
                    "x_min": anomaly.bbox.x_min,
                    "y_min": anomaly.bbox.y_min,
                    "x_max": anomaly.bbox.x_max,
                    "y_max": anomaly.bbox.y_max,
                },
                "geometry": {
                    "area_pixels": anomaly.geometry.area_pixels,
                    "perimeter_pixels": anomaly.geometry.perimeter_pixels,
                    "length_pixels": anomaly.geometry.length_pixels,
                    "width_pixels": anomaly.geometry.width_pixels,
                    "orientation_degrees": anomaly.geometry.orientation_degrees,
                    "aspect_ratio": anomaly.geometry.aspect_ratio,
                    "solidity": anomaly.geometry.solidity,
                    "centroid": anomaly.geometry.centroid,
                },
                "confidence": {
                    "detection": anomaly.detection_confidence,
                    "segmentation": anomaly.segmentation_confidence,
                    "association": anomaly.association_confidence,
                    "combined": anomaly.combined_confidence,
                },
            }

            # Include mask if requested
            if include_masks:
                anomaly_data["mask_base64"] = encode_mask(anomaly.mask.mask)

            anomalies.append(anomaly_data)

        structures = [
            {
                "detection_id": s.detection_id,
                "class_name": s.class_name,
                "confidence": s.confidence,
                "bbox": {
                    "x_min": s.bbox.x_min,
                    "y_min": s.bbox.y_min,
                    "x_max": s.bbox.x_max,
                    "y_max": s.bbox.y_max,
                },
            }
            for s in result.structures
        ]

        return {
            "frame_id": result.frame_id,
            "image_size": {
                "width": result.image_width,
                "height": result.image_height,
            },
            "anomalies": anomalies,
            "anomaly_count": len(anomalies),
            "structures": structures,
            "structure_count": len(structures),
            "processing_time_ms": result.processing_time_ms,
            "detector_time_ms": result.detector_time_ms,
            "segmenter_time_ms": result.segmenter_time_ms,
        }

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


# RunPod serverless entrypoint
if os.environ.get("RUNPOD_SERVERLESS", "false").lower() == "true":
    import runpod
    runpod.serverless.start({"handler": handler})
