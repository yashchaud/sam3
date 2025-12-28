"""
FastAPI Server for Pod Mode.

This module provides a REST API server for running the anomaly detection
pipeline in RunPod Pod mode (persistent GPU instance).

Endpoints:
    GET  /health       - Health check
    GET  /info         - Model and configuration info
    POST /detect       - Run anomaly detection on image
    POST /detect/batch - Run detection on multiple images

Environment Variables:
    DETECTOR_VARIANT: RF-DETR variant (nano, small, medium, large)
    DETECTOR_WEIGHTS: Path to custom detector weights (optional)
    DETECTOR_NUM_CLASSES: Number of classes for custom model (optional)
    SAM_MODEL_PATH: Path to SAM3/SAM2 model weights
    CONFIDENCE_THRESHOLD: Detection confidence threshold (default: 0.3)
    API_HOST: Server host (default: 0.0.0.0)
    API_PORT: Server port (default: 8000)
"""

import os
import base64
import time
from pathlib import Path
from io import BytesIO
from typing import Optional, List
from contextlib import asynccontextmanager

import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Global pipeline instance
pipeline = None
load_time = None


class DetectionRequest(BaseModel):
    """Request model for detection endpoint."""
    image: str = Field(..., description="Base64 encoded image")
    frame_id: Optional[str] = Field(None, description="Optional frame identifier")
    include_masks: bool = Field(False, description="Include masks in response")
    confidence_threshold: Optional[float] = Field(
        None, description="Override confidence threshold"
    )


class URLDetectionRequest(BaseModel):
    """Request model for URL detection endpoint."""
    image_url: str = Field(..., description="URL of image to process")
    frame_id: Optional[str] = Field(None, description="Optional frame identifier")
    include_masks: bool = Field(False, description="Include masks in response")
    confidence_threshold: Optional[float] = Field(
        None, description="Override confidence threshold"
    )


class BatchDetectionRequest(BaseModel):
    """Request model for batch detection endpoint."""
    images: List[str] = Field(..., description="List of base64 encoded images")
    frame_ids: Optional[List[str]] = Field(None, description="Optional frame identifiers")
    include_masks: bool = Field(False, description="Include masks in response")


class BoundingBoxResponse(BaseModel):
    x_min: float
    y_min: float
    x_max: float
    y_max: float


class GeometryResponse(BaseModel):
    area_pixels: int
    perimeter_pixels: float
    length_pixels: float
    width_pixels: float
    orientation_degrees: float
    aspect_ratio: float
    solidity: float
    centroid: tuple


class ConfidenceResponse(BaseModel):
    detection: float
    segmentation: float
    association: Optional[float]
    combined: float


class AnomalyResponse(BaseModel):
    anomaly_id: str
    defect_type: str
    structure_type: Optional[str]
    structure_id: Optional[str]
    bbox: BoundingBoxResponse
    geometry: GeometryResponse
    confidence: ConfidenceResponse
    mask_base64: Optional[str] = None


class StructureResponse(BaseModel):
    detection_id: str
    class_name: str
    confidence: float
    bbox: BoundingBoxResponse


class DetectionResponse(BaseModel):
    frame_id: str
    image_size: dict
    anomalies: List[AnomalyResponse]
    anomaly_count: int
    structures: List[StructureResponse]
    structure_count: int
    processing_time_ms: float
    detector_time_ms: float
    segmenter_time_ms: float


def load_pipeline_instance():
    """Load the pipeline."""
    global pipeline, load_time

    from anomaly_detection.pipeline import Phase1Pipeline, PipelineConfig, RFDETRVariant

    start = time.time()

    # Get configuration from environment
    variant_name = os.environ.get("DETECTOR_VARIANT", "medium").upper()
    variant = getattr(RFDETRVariant, variant_name, RFDETRVariant.MEDIUM)

    detector_weights = os.environ.get("DETECTOR_WEIGHTS")
    detector_num_classes = os.environ.get("DETECTOR_NUM_CLASSES")
    sam_model_path = os.environ.get("SAM_MODEL_PATH", "/models/sam3.pt")
    confidence = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.3"))

    config = PipelineConfig(
        segmenter_model_path=Path(sam_model_path),
        detector_variant=variant,
        detector_weights=Path(detector_weights) if detector_weights else None,
        detector_num_classes=int(detector_num_classes) if detector_num_classes else None,
        device="cuda",
    )

    if config.detector_config:
        config.detector_config.confidence_threshold = confidence

    pipeline = Phase1Pipeline(config)
    pipeline.load()

    load_time = time.time() - start
    print(f"Pipeline loaded in {load_time:.2f}s: RF-DETR {variant.value} + SAM3")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - load models on startup."""
    load_pipeline_instance()
    yield
    if pipeline:
        pipeline.unload()


app = FastAPI(
    title="Anomaly Detection API",
    description="Phase-1 Visual Anomaly Detection using RF-DETR + SAM3",
    version="0.1.0",
    lifespan=lifespan,
)


def decode_image(image_data: str) -> np.ndarray:
    """Decode base64 image to numpy array."""
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


def process_result(result, include_masks: bool = False) -> dict:
    """Convert pipeline result to response dict."""
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


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "pipeline_loaded": pipeline is not None and pipeline.is_loaded(),
    }


@app.get("/info")
async def get_info():
    """Get model and configuration info."""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not loaded")

    return {
        "detector": {
            "type": "RF-DETR",
            "variant": pipeline.config.detector_variant.value,
            "confidence_threshold": pipeline.config.detector_config.confidence_threshold,
        },
        "segmenter": {
            "type": "SAM3",
            "model_path": str(pipeline.config.segmenter_model_path),
        },
        "load_time_seconds": load_time,
    }


@app.post("/detect")
async def detect(request: DetectionRequest):
    """
    Run anomaly detection on a single image.

    Returns detected anomalies with bounding boxes, masks, and geometry.
    """
    if not pipeline or not pipeline.is_loaded():
        raise HTTPException(status_code=503, detail="Pipeline not loaded")

    try:
        image = decode_image(request.image)

        # Override confidence if provided
        original_conf = None
        if request.confidence_threshold is not None:
            original_conf = pipeline.config.detector_config.confidence_threshold
            pipeline.config.detector_config.confidence_threshold = request.confidence_threshold

        result = pipeline.process(image, frame_id=request.frame_id)

        # Restore confidence
        if original_conf is not None:
            pipeline.config.detector_config.confidence_threshold = original_conf

        return process_result(result, request.include_masks)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect/batch")
async def detect_batch(request: BatchDetectionRequest):
    """
    Run anomaly detection on multiple images.

    Returns list of detection results, one per image.
    """
    if not pipeline or not pipeline.is_loaded():
        raise HTTPException(status_code=503, detail="Pipeline not loaded")

    try:
        images = [decode_image(img) for img in request.images]
        frame_ids = request.frame_ids

        results = pipeline.process_batch(images, frame_ids=frame_ids)

        return {
            "results": [
                process_result(r, request.include_masks)
                for r in results
            ],
            "count": len(results),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect/upload")
async def detect_upload(
    file: UploadFile = File(...),
    include_masks: bool = False,
):
    """
    Run anomaly detection on uploaded image file.
    """
    if not pipeline or not pipeline.is_loaded():
        raise HTTPException(status_code=503, detail="Pipeline not loaded")

    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        image_array = np.array(image)

        result = pipeline.process(image_array, frame_id=file.filename)

        return process_result(result, include_masks)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect/url")
async def detect_url(request: URLDetectionRequest):
    """
    Run anomaly detection on image from URL.
    """
    if not pipeline or not pipeline.is_loaded():
        raise HTTPException(status_code=503, detail="Pipeline not loaded")

    try:
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.get(request.image_url)
            response.raise_for_status()

        image = Image.open(BytesIO(response.content)).convert("RGB")
        image_array = np.array(image)

        # Override confidence if provided
        original_conf = None
        if request.confidence_threshold is not None:
            original_conf = pipeline.config.detector_config.confidence_threshold
            pipeline.config.detector_config.confidence_threshold = request.confidence_threshold

        result = pipeline.process(image_array, frame_id=request.frame_id)

        # Restore confidence
        if original_conf is not None:
            pipeline.config.detector_config.confidence_threshold = original_conf

        return process_result(result, request.include_masks)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Aliases for /process/* routes (for compatibility)
app.post("/process")(detect)
app.post("/process/upload")(detect_upload)
app.post("/process/url")(detect_url)
app.post("/batch/process")(detect_batch)


def main():
    """Run the API server."""
    host = os.environ.get("API_HOST", "0.0.0.0")
    port = int(os.environ.get("API_PORT", "8000"))

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
