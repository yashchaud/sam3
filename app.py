from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query, Body, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, JSONResponse, StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
import io
import base64
import numpy as np
from typing import Optional, List, Union
import logging
import traceback
from pydantic import BaseModel, Field, HttpUrl
import cv2
import httpx
import os
import uuid
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SAM3 Segmentation API",
    description="Complete API service for SAM3 image and video segmentation with file and base64 image support",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for the web interface
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    logger.info(f"Static files mounted from {static_dir}")
else:
    logger.warning(f"Static directory not found: {static_dir}")

# Global variables
model = None
processor = None
device = None
model_loaded = False

# Tracker model for point-based prompts (Transformers API)
tracker_model = None
tracker_processor = None

# Real-time session management
realtime_sessions = {}  # {session_id: {"inference_state": state, "image": pil_image, "last_activity": datetime}}

# Pydantic models for JSON requests
class TextSegmentRequest(BaseModel):
    image_base64: str = Field(..., description="Base64-encoded image")
    text: str = Field(..., description="Text description of objects to segment")
    return_visualization: bool = Field(False, description="Return visualization image")
    mask_threshold: float = Field(0.5, ge=0.0, le=1.0, description="Mask confidence threshold")

class TextSegmentURLRequest(BaseModel):
    image_url: str = Field(..., description="URL of the image to segment")
    text: str = Field(..., description="Text description of objects to segment")
    return_visualization: bool = Field(False, description="Return visualization image")
    mask_threshold: float = Field(0.5, ge=0.0, le=1.0, description="Mask confidence threshold")

class BoxSegmentRequest(BaseModel):
    image_base64: str = Field(..., description="Base64-encoded image")
    x1: int = Field(..., description="Top-left x coordinate")
    y1: int = Field(..., description="Top-left y coordinate")
    x2: int = Field(..., description="Bottom-right x coordinate")
    y2: int = Field(..., description="Bottom-right y coordinate")
    return_visualization: bool = Field(False, description="Return visualization image")

class BoxSegmentURLRequest(BaseModel):
    image_url: str = Field(..., description="URL of the image to segment")
    x1: int = Field(..., description="Top-left x coordinate")
    y1: int = Field(..., description="Top-left y coordinate")
    x2: int = Field(..., description="Bottom-right x coordinate")
    y2: int = Field(..., description="Bottom-right y coordinate")
    return_visualization: bool = Field(False, description="Return visualization image")

class PointSegmentRequest(BaseModel):
    image_base64: str = Field(..., description="Base64-encoded image")
    points_x: List[int] = Field(..., description="List of x coordinates")
    points_y: List[int] = Field(..., description="List of y coordinates")
    labels: List[int] = Field(..., description="List of labels (1=positive, 0=negative)")
    return_visualization: bool = Field(False, description="Return visualization image")

class PointSegmentURLRequest(BaseModel):
    image_url: str = Field(..., description="URL of the image to segment")
    points_x: List[int] = Field(..., description="List of x coordinates")
    points_y: List[int] = Field(..., description="List of y coordinates")
    labels: List[int] = Field(..., description="List of labels (1=positive, 0=negative)")
    return_visualization: bool = Field(False, description="Return visualization image")

class AutoSegmentRequest(BaseModel):
    image_base64: str = Field(..., description="Base64-encoded image")
    return_visualization: bool = Field(False, description="Return visualization image")
    points_per_side: int = Field(32, ge=1, le=64, description="Points per side for grid")
    pred_iou_thresh: float = Field(0.88, ge=0.0, le=1.0, description="IoU threshold")
    stability_score_thresh: float = Field(0.95, ge=0.0, le=1.0, description="Stability score threshold")

class AutoSegmentURLRequest(BaseModel):
    image_url: str = Field(..., description="URL of the image to segment")
    return_visualization: bool = Field(False, description="Return visualization image")
    points_per_side: int = Field(32, ge=1, le=64, description="Points per side for grid")
    pred_iou_thresh: float = Field(0.88, ge=0.0, le=1.0, description="IoU threshold")
    stability_score_thresh: float = Field(0.95, ge=0.0, le=1.0, description="Stability score threshold")

class SegmentationResponse(BaseModel):
    success: bool
    num_masks: int
    masks: List[str]
    boxes: Optional[List[List[float]]] = None
    scores: Optional[List[float]] = None
    prompt: Optional[str] = None
    message: Optional[str] = None

class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    detail: Optional[str] = None

@app.on_event("startup")
async def load_model():
    """Load the SAM3 model on startup"""
    global model, processor, device, model_loaded

    try:
        logger.info("Loading SAM3 model...")

        # Import SAM3 - mock training dependencies first
        try:
            import sys
            import torch

            logger.info("Setting up module mocks for training dependencies...")

             # These are only used in training code, not for inference
            class MockModule:
                def __init__(self, name="MockModule"):
                    self._name = name

                def __getattr__(self, name):
                    # Return a new mock for any attribute access
                    return MockModule(f"{self._name}.{name}")

                def __call__(self, *args, **kwargs):
                    # Allow the mock to be called
                    return MockModule(f"{self._name}()")

            # Mock all training-only dependencies
            sys.modules['decord'] = MockModule('decord')
            sys.modules['pycocotools'] = MockModule('pycocotools')
            sys.modules['pycocotools.mask'] = MockModule('pycocotools.mask')

            logger.info("Mocks installed successfully")
            logger.info("Importing SAM3 components...")

            # Import native SAM3 for text/box prompts
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor

            # Import Transformers SAM3Tracker for point prompts
            from transformers import Sam3TrackerProcessor, Sam3TrackerModel

            logger.info("SAM3 modules imported successfully")

            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device}")

            # Build native SAM3 model for text/box prompts
            logger.info("Building native SAM3 model...")
            model = build_sam3_image_model()
            logger.info("Model built, moving to device...")

            if device == "cuda":
                model = model.cuda()

            logger.info("Creating SAM3 processor...")
            processor = Sam3Processor(model)

            # Load Transformers SAM3Tracker for point-based prompts
            logger.info("Loading SAM3 Tracker for point prompts...")
            global tracker_model, tracker_processor
            tracker_model = Sam3TrackerModel.from_pretrained("facebook/sam3").to(device)
            tracker_processor = Sam3TrackerProcessor.from_pretrained("facebook/sam3")

            model_loaded = True

            logger.info("SAM3 models loaded successfully!")

        except ImportError as e:
            logger.error(f"Failed to import SAM3: {e}")
            logger.error("Please install SAM3 following the instructions in README.md")
            logger.error(traceback.format_exc())
            model_loaded = False
        except Exception as e:
            logger.error(f"Error during SAM3 import/setup: {e}")
            logger.error(traceback.format_exc())
            model_loaded = False

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        logger.error(traceback.format_exc())
        model_loaded = False

def check_model_loaded():
    """Check if model is loaded and raise error if not"""
    if not model_loaded:
        raise HTTPException(
            status_code=503,
            detail="SAM3 model not loaded. Please check server logs and ensure SAM3 is properly installed."
        )

async def download_image_from_url(image_url: str) -> bytes:
    """Download image from URL"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(image_url)

            if response.status_code != 200:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to download image from URL. Status code: {response.status_code}"
                )

            # Check content type
            content_type = response.headers.get('content-type', '')
            if not content_type.startswith('image/'):
                logger.warning(f"URL content-type is '{content_type}', expected image/*")

            # Check file size (limit to 50MB)
            content_length = len(response.content)
            if content_length > 50 * 1024 * 1024:
                raise HTTPException(
                    status_code=400,
                    detail=f"Image too large ({content_length / 1024 / 1024:.1f}MB). Maximum size is 50MB."
                )

            logger.info(f"Downloaded image from URL: {image_url} ({content_length / 1024:.1f}KB)")
            return response.content

    except httpx.TimeoutException:
        raise HTTPException(
            status_code=408,
            detail="Timeout while downloading image from URL"
        )
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to download image from URL: {str(e)}"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error downloading image: {str(e)}"
        )

def decode_base64_image(base64_string: str) -> bytes:
    """Decode base64 image string to bytes"""
    try:
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',', 1)[1]

        # Decode base64
        image_bytes = base64.b64decode(base64_string)
        return image_bytes
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid base64 image string: {str(e)}"
        )

def validate_image(image_data: bytes) -> Image.Image:
    """Validate and load image"""
    try:
        pil_image = Image.open(io.BytesIO(image_data))

        # Convert to RGB if needed
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')

        # Check image size
        if pil_image.size[0] > 4096 or pil_image.size[1] > 4096:
            logger.warning(f"Large image detected: {pil_image.size}. Resizing...")
            pil_image.thumbnail((4096, 4096), Image.Resampling.LANCZOS)

        return pil_image

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image file: {str(e)}"
        )

def masks_to_base64(masks: np.ndarray) -> List[str]:
    """Convert masks to base64-encoded PNG images"""
    masks_base64 = []

    try:
        # Handle different mask formats
        if len(masks.shape) == 2:
            masks = masks[np.newaxis, ...]

        for mask in masks:
            # Convert to uint8
            if mask.dtype == bool:
                mask_array = mask.astype(np.uint8) * 255
            else:
                mask_array = (mask * 255).astype(np.uint8)

            # Create PIL image
            mask_image = Image.fromarray(mask_array)

            # Encode to PNG
            buffer = io.BytesIO()
            mask_image.save(buffer, format="PNG")
            mask_b64 = base64.b64encode(buffer.getvalue()).decode()
            masks_base64.append(mask_b64)

    except Exception as e:
        logger.error(f"Error encoding masks: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error encoding masks: {str(e)}"
        )

    return masks_base64

def create_visualization(image: Image.Image, masks: np.ndarray, boxes=None) -> bytes:
    """Create visualization with masks overlaid on image"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image)

        # Overlay masks
        if len(masks.shape) == 2:
            masks = masks[np.newaxis, ...]

        for i, mask in enumerate(masks):
            color = np.random.random(3)
            # Ensure mask is 2D
            if len(mask.shape) > 2:
                # Flatten extra dimensions
                mask = mask.squeeze()

            # Get the actual height and width from the mask
            if len(mask.shape) == 2:
                h, w = mask.shape
            else:
                # If still not 2D, something is wrong - log and skip
                logger.warning(f"Unexpected mask shape: {mask.shape}, skipping")
                continue

            # Create colored mask overlay
            mask_image = mask[:, :, np.newaxis] * color.reshape(1, 1, -1)
            ax.imshow(mask_image, alpha=0.5)

        # Draw boxes if provided
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box
                rect = patches.Rectangle(
                    (x1, y1), x2-x1, y2-y1,
                    linewidth=2, edgecolor='red', facecolor='none'
                )
                ax.add_patch(rect)

        ax.axis('off')
        plt.tight_layout()

        # Save to buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format='PNG', bbox_inches='tight', dpi=150)
        plt.close(fig)
        buffer.seek(0)

        return buffer.getvalue()

    except Exception as e:
        logger.error(f"Error creating visualization: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error creating visualization: {str(e)}"
        )

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "model": "SAM3",
        "model_loaded": model_loaded,
        "device": device if model_loaded else "N/A",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    check_model_loaded()
    return {
        "status": "healthy",
        "model_loaded": True,
        "device": device
    }

@app.get("/app", response_class=HTMLResponse)
async def serve_app():
    """Serve the real-time segmentation web interface"""
    index_path = os.path.join(static_dir, "index.html")

    if os.path.exists(index_path):
        with open(index_path, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        return "<h1>Error: Web interface not found</h1><p>Please ensure static/index.html exists.</p>"

# File upload endpoints
@app.post("/segment/text", response_model=SegmentationResponse)
async def segment_with_text(
    image: UploadFile = File(..., description="Image file to segment"),
    text: str = Form(..., description="Text description of objects to segment"),
    return_visualization: bool = Form(False, description="Return visualization image instead of masks"),
    mask_threshold: float = Form(0.5, ge=0.0, le=1.0, description="Mask confidence threshold")
):
    """
    Segment objects in an image using a text prompt (file upload).

    This endpoint supports open-vocabulary segmentation using natural language descriptions.
    """
    check_model_loaded()

    try:
        # Validate and load image
        image_data = await image.read()
        pil_image = validate_image(image_data)

        logger.info(f"Processing image with text prompt: '{text}'")

        # Set image in processor
        inference_state = processor.set_image(pil_image)

        # Run text-based segmentation
        output = processor.set_text_prompt(
            state=inference_state,
            prompt=text
        )

        # Extract results
        masks = output.get("masks")
        boxes = output.get("boxes")
        scores = output.get("scores")

        if masks is None or len(masks) == 0:
            return SegmentationResponse(
                success=True,
                num_masks=0,
                masks=[],
                message=f"No objects found matching '{text}'"
            )

        # Convert masks to numpy if needed
        if hasattr(masks, 'cpu'):
            masks = masks.cpu().numpy()
        if hasattr(boxes, 'cpu'):
            boxes = boxes.cpu().numpy()
        if hasattr(scores, 'cpu'):
            scores = scores.cpu().numpy()

        # Return visualization if requested
        if return_visualization:
            viz_image = create_visualization(pil_image, masks, boxes)
            return Response(content=viz_image, media_type="image/png")

        # Convert masks to base64
        masks_base64 = masks_to_base64(masks)

        return SegmentationResponse(
            success=True,
            num_masks=len(masks_base64),
            masks=masks_base64,
            boxes=boxes.tolist() if boxes is not None else None,
            scores=scores.tolist() if scores is not None else None,
            prompt=text
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in text segmentation: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Segmentation failed: {str(e)}"
        )

# Base64 JSON endpoints
@app.post("/segment/text/base64", response_model=SegmentationResponse)
async def segment_with_text_base64(request: TextSegmentRequest):
    """
    Segment objects in an image using a text prompt (base64 input).

    This endpoint accepts base64-encoded images in JSON format.
    """
    check_model_loaded()

    try:
        # Decode base64 image
        image_data = decode_base64_image(request.image_base64)
        pil_image = validate_image(image_data)

        logger.info(f"Processing base64 image with text prompt: '{request.text}'")

        # Set image in processor
        inference_state = processor.set_image(pil_image)

        # Run text-based segmentation
        output = processor.set_text_prompt(
            state=inference_state,
            prompt=request.text
        )

        # Extract results
        masks = output.get("masks")
        boxes = output.get("boxes")
        scores = output.get("scores")

        if masks is None or len(masks) == 0:
            return SegmentationResponse(
                success=True,
                num_masks=0,
                masks=[],
                message=f"No objects found matching '{request.text}'"
            )

        # Convert masks to numpy if needed
        if hasattr(masks, 'cpu'):
            masks = masks.cpu().numpy()
        if hasattr(boxes, 'cpu'):
            boxes = boxes.cpu().numpy()
        if hasattr(scores, 'cpu'):
            scores = scores.cpu().numpy()

        # Return visualization if requested
        if request.return_visualization:
            viz_image = create_visualization(pil_image, masks, boxes)
            viz_base64 = base64.b64encode(viz_image).decode()
            return SegmentationResponse(
                success=True,
                num_masks=1,
                masks=[viz_base64],
                message="Visualization returned in masks[0]"
            )

        # Convert masks to base64
        masks_base64 = masks_to_base64(masks)

        return SegmentationResponse(
            success=True,
            num_masks=len(masks_base64),
            masks=masks_base64,
            boxes=boxes.tolist() if boxes is not None else None,
            scores=scores.tolist() if scores is not None else None,
            prompt=request.text
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in text segmentation: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Segmentation failed: {str(e)}"
        )

@app.post("/segment/box", response_model=SegmentationResponse)
async def segment_with_box(
    image: UploadFile = File(..., description="Image file to segment"),
    x1: int = Form(..., description="Top-left x coordinate"),
    y1: int = Form(..., description="Top-left y coordinate"),
    x2: int = Form(..., description="Bottom-right x coordinate"),
    y2: int = Form(..., description="Bottom-right y coordinate"),
    return_visualization: bool = Form(False, description="Return visualization image")
):
    """
    Segment objects within a bounding box (file upload).

    Provide coordinates of a bounding box to segment the object within it.
    """
    check_model_loaded()

    try:
        # Validate box coordinates
        if x1 >= x2 or y1 >= y2:
            raise HTTPException(
                status_code=400,
                detail="Invalid bounding box: x1 must be < x2 and y1 must be < y2"
            )

        # Load image
        image_data = await image.read()
        pil_image = validate_image(image_data)

        # Validate box is within image bounds
        w, h = pil_image.size
        if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
            raise HTTPException(
                status_code=400,
                detail=f"Bounding box out of image bounds. Image size: {w}x{h}"
            )

        logger.info(f"Processing image with box: [{x1}, {y1}, {x2}, {y2}]")

        # Set image in processor
        inference_state = processor.set_image(pil_image)

        # Add box prompt
        output = processor.add_geometric_prompt(
            state=inference_state,
            boxes=[[x1, y1, x2, y2]]
        )

        # Extract results
        masks = output.get("masks")
        scores = output.get("scores")

        if masks is None or len(masks) == 0:
            return SegmentationResponse(
                success=True,
                num_masks=0,
                masks=[],
                message="No objects found in the specified box"
            )

        # Convert to numpy
        if hasattr(masks, 'cpu'):
            masks = masks.cpu().numpy()
        if hasattr(scores, 'cpu'):
            scores = scores.cpu().numpy()

        # Return visualization if requested
        if return_visualization:
            viz_image = create_visualization(pil_image, masks, [[x1, y1, x2, y2]])
            return Response(content=viz_image, media_type="image/png")

        # Convert masks to base64
        masks_base64 = masks_to_base64(masks)

        return SegmentationResponse(
            success=True,
            num_masks=len(masks_base64),
            masks=masks_base64,
            boxes=[[x1, y1, x2, y2]],
            scores=scores.tolist() if scores is not None else None
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in box segmentation: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Segmentation failed: {str(e)}"
        )

@app.post("/segment/box/base64", response_model=SegmentationResponse)
async def segment_with_box_base64(request: BoxSegmentRequest):
    """
    Segment objects within a bounding box (base64 input).

    This endpoint accepts base64-encoded images in JSON format.
    """
    check_model_loaded()

    try:
        # Validate box coordinates
        if request.x1 >= request.x2 or request.y1 >= request.y2:
            raise HTTPException(
                status_code=400,
                detail="Invalid bounding box: x1 must be < x2 and y1 must be < y2"
            )

        # Decode base64 image
        image_data = decode_base64_image(request.image_base64)
        pil_image = validate_image(image_data)

        # Validate box is within image bounds
        w, h = pil_image.size
        if request.x1 < 0 or request.y1 < 0 or request.x2 > w or request.y2 > h:
            raise HTTPException(
                status_code=400,
                detail=f"Bounding box out of image bounds. Image size: {w}x{h}"
            )

        logger.info(f"Processing base64 image with box: [{request.x1}, {request.y1}, {request.x2}, {request.y2}]")

        # Set image in processor
        inference_state = processor.set_image(pil_image)

        # Add box prompt
        output = processor.add_geometric_prompt(
            state=inference_state,
            boxes=[[request.x1, request.y1, request.x2, request.y2]]
        )

        # Extract results
        masks = output.get("masks")
        scores = output.get("scores")

        if masks is None or len(masks) == 0:
            return SegmentationResponse(
                success=True,
                num_masks=0,
                masks=[],
                message="No objects found in the specified box"
            )

        # Convert to numpy
        if hasattr(masks, 'cpu'):
            masks = masks.cpu().numpy()
        if hasattr(scores, 'cpu'):
            scores = scores.cpu().numpy()

        # Return visualization if requested
        if request.return_visualization:
            viz_image = create_visualization(pil_image, masks, [[request.x1, request.y1, request.x2, request.y2]])
            viz_base64 = base64.b64encode(viz_image).decode()
            return SegmentationResponse(
                success=True,
                num_masks=1,
                masks=[viz_base64],
                message="Visualization returned in masks[0]"
            )

        # Convert masks to base64
        masks_base64 = masks_to_base64(masks)

        return SegmentationResponse(
            success=True,
            num_masks=len(masks_base64),
            masks=masks_base64,
            boxes=[[request.x1, request.y1, request.x2, request.y2]],
            scores=scores.tolist() if scores is not None else None
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in box segmentation: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Segmentation failed: {str(e)}"
        )

@app.post("/segment/points", response_model=SegmentationResponse)
async def segment_with_points(
    image: UploadFile = File(..., description="Image file to segment"),
    points_x: str = Form(..., description="Comma-separated x coordinates"),
    points_y: str = Form(..., description="Comma-separated y coordinates"),
    labels: str = Form(..., description="Comma-separated labels (1=positive, 0=negative)"),
    return_visualization: bool = Form(False, description="Return visualization image")
):
    """
    Segment objects using point prompts (file upload).

    Points can be positive (on the object) or negative (not on the object).
    Labels: 1 = positive point, 0 = negative point
    """
    check_model_loaded()

    try:
        # Parse points
        try:
            x_coords = [int(x.strip()) for x in points_x.split(",")]
            y_coords = [int(y.strip()) for y in points_y.split(",")]
            point_labels = [int(l.strip()) for l in labels.split(",")]
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid point coordinates or labels: {str(e)}"
            )

        # Validate same length
        if not (len(x_coords) == len(y_coords) == len(point_labels)):
            raise HTTPException(
                status_code=400,
                detail="Points X, Y, and labels must have the same length"
            )

        # Validate labels are 0 or 1
        if not all(l in [0, 1] for l in point_labels):
            raise HTTPException(
                status_code=400,
                detail="Labels must be 0 (negative) or 1 (positive)"
            )

        # Load image
        image_data = await image.read()
        pil_image = validate_image(image_data)

        # Validate points are within image
        w, h = pil_image.size
        for x, y in zip(x_coords, y_coords):
            if x < 0 or y < 0 or x >= w or y >= h:
                raise HTTPException(
                    status_code=400,
                    detail=f"Point ({x}, {y}) is out of image bounds ({w}x{h})"
                )

        logger.info(f"Processing image with {len(x_coords)} points")

        # Set image in processor
        inference_state = processor.set_image(pil_image)

        # Prepare points
        points = [[x, y] for x, y in zip(x_coords, y_coords)]

        # Add point prompts
        output = processor.add_geometric_prompt(
            state=inference_state,
            points=points,
            labels=point_labels
        )

        # Extract results
        masks = output.get("masks")
        scores = output.get("scores")

        if masks is None or len(masks) == 0:
            return SegmentationResponse(
                success=True,
                num_masks=0,
                masks=[],
                message="No objects found at the specified points"
            )

        # Convert to numpy
        if hasattr(masks, 'cpu'):
            masks = masks.cpu().numpy()
        if hasattr(scores, 'cpu'):
            scores = scores.cpu().numpy()

        # Return visualization if requested
        if return_visualization:
            viz_image = create_visualization(pil_image, masks)
            return Response(content=viz_image, media_type="image/png")

        # Convert masks to base64
        masks_base64 = masks_to_base64(masks)

        return SegmentationResponse(
            success=True,
            num_masks=len(masks_base64),
            masks=masks_base64,
            scores=scores.tolist() if scores is not None else None
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in point segmentation: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Segmentation failed: {str(e)}"
        )

@app.post("/segment/points/base64", response_model=SegmentationResponse)
async def segment_with_points_base64(request: PointSegmentRequest):
    """
    Segment objects using point prompts (base64 input).

    This endpoint accepts base64-encoded images in JSON format.
    """
    check_model_loaded()

    try:
        # Validate same length
        if not (len(request.points_x) == len(request.points_y) == len(request.labels)):
            raise HTTPException(
                status_code=400,
                detail="Points X, Y, and labels must have the same length"
            )

        # Validate labels are 0 or 1
        if not all(l in [0, 1] for l in request.labels):
            raise HTTPException(
                status_code=400,
                detail="Labels must be 0 (negative) or 1 (positive)"
            )

        # Decode base64 image
        image_data = decode_base64_image(request.image_base64)
        pil_image = validate_image(image_data)

        # Validate points are within image
        w, h = pil_image.size
        for x, y in zip(request.points_x, request.points_y):
            if x < 0 or y < 0 or x >= w or y >= h:
                raise HTTPException(
                    status_code=400,
                    detail=f"Point ({x}, {y}) is out of image bounds ({w}x{h})"
                )

        logger.info(f"Processing base64 image with {len(request.points_x)} points")

        # Set image in processor
        inference_state = processor.set_image(pil_image)

        # Prepare points
        points = [[x, y] for x, y in zip(request.points_x, request.points_y)]

        # Add point prompts
        output = processor.add_geometric_prompt(
            state=inference_state,
            points=points,
            labels=request.labels
        )

        # Extract results
        masks = output.get("masks")
        scores = output.get("scores")

        if masks is None or len(masks) == 0:
            return SegmentationResponse(
                success=True,
                num_masks=0,
                masks=[],
                message="No objects found at the specified points"
            )

        # Convert to numpy
        if hasattr(masks, 'cpu'):
            masks = masks.cpu().numpy()
        if hasattr(scores, 'cpu'):
            scores = scores.cpu().numpy()

        # Return visualization if requested
        if request.return_visualization:
            viz_image = create_visualization(pil_image, masks)
            viz_base64 = base64.b64encode(viz_image).decode()
            return SegmentationResponse(
                success=True,
                num_masks=1,
                masks=[viz_base64],
                message="Visualization returned in masks[0]"
            )

        # Convert masks to base64
        masks_base64 = masks_to_base64(masks)

        return SegmentationResponse(
            success=True,
            num_masks=len(masks_base64),
            masks=masks_base64,
            scores=scores.tolist() if scores is not None else None
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in point segmentation: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Segmentation failed: {str(e)}"
        )

@app.post("/segment/auto")
async def segment_automatic(
    image: UploadFile = File(..., description="Image file to segment"),
    return_visualization: bool = Form(False, description="Return visualization image"),
    points_per_side: int = Form(32, ge=1, le=64, description="Number of points per side for grid"),
    pred_iou_thresh: float = Form(0.88, ge=0.0, le=1.0, description="IoU threshold"),
    stability_score_thresh: float = Form(0.95, ge=0.0, le=1.0, description="Stability score threshold")
):
    """
    Automatically segment all objects in an image (file upload).

    This endpoint generates masks for all objects without requiring prompts.
    """
    check_model_loaded()

    try:
        # Load image
        image_data = await image.read()
        pil_image = validate_image(image_data)

        logger.info("Processing image with automatic segmentation")

        # Set image in processor
        inference_state = processor.set_image(pil_image)

        # Run automatic segmentation
        output = processor.generate_masks(
            state=inference_state,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh
        )

        # Extract results
        masks = output.get("masks")
        scores = output.get("scores")

        if masks is None or len(masks) == 0:
            return SegmentationResponse(
                success=True,
                num_masks=0,
                masks=[],
                message="No objects found in automatic segmentation"
            )

        # Convert to numpy
        if hasattr(masks, 'cpu'):
            masks = masks.cpu().numpy()
        if hasattr(scores, 'cpu'):
            scores = scores.cpu().numpy()

        logger.info(f"Found {len(masks)} objects")

        # Return visualization if requested
        if return_visualization:
            viz_image = create_visualization(pil_image, masks)
            return Response(content=viz_image, media_type="image/png")

        # Convert masks to base64
        masks_base64 = masks_to_base64(masks)

        return SegmentationResponse(
            success=True,
            num_masks=len(masks_base64),
            masks=masks_base64,
            scores=scores.tolist() if scores is not None else None
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in automatic segmentation: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Segmentation failed: {str(e)}"
        )

@app.post("/segment/auto/base64", response_model=SegmentationResponse)
async def segment_automatic_base64(request: AutoSegmentRequest):
    """
    Automatically segment all objects in an image (base64 input).

    This endpoint accepts base64-encoded images in JSON format.
    """
    check_model_loaded()

    try:
        # Decode base64 image
        image_data = decode_base64_image(request.image_base64)
        pil_image = validate_image(image_data)

        logger.info("Processing base64 image with automatic segmentation")

        # Set image in processor
        inference_state = processor.set_image(pil_image)

        # Run automatic segmentation
        output = processor.generate_masks(
            state=inference_state,
            points_per_side=request.points_per_side,
            pred_iou_thresh=request.pred_iou_thresh,
            stability_score_thresh=request.stability_score_thresh
        )

        # Extract results
        masks = output.get("masks")
        scores = output.get("scores")

        if masks is None or len(masks) == 0:
            return SegmentationResponse(
                success=True,
                num_masks=0,
                masks=[],
                message="No objects found in automatic segmentation"
            )

        # Convert to numpy
        if hasattr(masks, 'cpu'):
            masks = masks.cpu().numpy()
        if hasattr(scores, 'cpu'):
            scores = scores.cpu().numpy()

        logger.info(f"Found {len(masks)} objects")

        # Return visualization if requested
        if request.return_visualization:
            viz_image = create_visualization(pil_image, masks)
            viz_base64 = base64.b64encode(viz_image).decode()
            return SegmentationResponse(
                success=True,
                num_masks=1,
                masks=[viz_base64],
                message="Visualization returned in masks[0]"
            )

        # Convert masks to base64
        masks_base64 = masks_to_base64(masks)

        return SegmentationResponse(
            success=True,
            num_masks=len(masks_base64),
            masks=masks_base64,
            scores=scores.tolist() if scores is not None else None
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in automatic segmentation: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Segmentation failed: {str(e)}"
        )

# URL-based endpoints
@app.post("/segment/text/url", response_model=SegmentationResponse)
async def segment_with_text_url(request: TextSegmentURLRequest):
    """
    Segment objects in an image using a text prompt (URL input).

    This endpoint accepts an image URL and downloads it for processing.
    """
    check_model_loaded()

    try:
        # Download image from URL
        image_data = await download_image_from_url(request.image_url)
        pil_image = validate_image(image_data)

        logger.info(f"Processing URL image with text prompt: '{request.text}'")

        # Set image in processor
        inference_state = processor.set_image(pil_image)

        # Run text-based segmentation
        output = processor.set_text_prompt(
            state=inference_state,
            prompt=request.text
        )

        # Extract results
        masks = output.get("masks")
        boxes = output.get("boxes")
        scores = output.get("scores")

        if masks is None or len(masks) == 0:
            return SegmentationResponse(
                success=True,
                num_masks=0,
                masks=[],
                message=f"No objects found matching '{request.text}'"
            )

        # Convert masks to numpy if needed
        if hasattr(masks, 'cpu'):
            masks = masks.cpu().numpy()
        if hasattr(boxes, 'cpu'):
            boxes = boxes.cpu().numpy()
        if hasattr(scores, 'cpu'):
            scores = scores.cpu().numpy()

        # Return visualization if requested
        if request.return_visualization:
            viz_image = create_visualization(pil_image, masks, boxes)
            viz_base64 = base64.b64encode(viz_image).decode()
            return SegmentationResponse(
                success=True,
                num_masks=1,
                masks=[viz_base64],
                message="Visualization returned in masks[0]"
            )

        # Convert masks to base64
        masks_base64 = masks_to_base64(masks)

        return SegmentationResponse(
            success=True,
            num_masks=len(masks_base64),
            masks=masks_base64,
            boxes=boxes.tolist() if boxes is not None else None,
            scores=scores.tolist() if scores is not None else None,
            prompt=request.text
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in text segmentation: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Segmentation failed: {str(e)}"
        )

@app.post("/segment/box/url", response_model=SegmentationResponse)
async def segment_with_box_url(request: BoxSegmentURLRequest):
    """
    Segment objects within a bounding box (URL input).

    This endpoint accepts an image URL and downloads it for processing.
    """
    check_model_loaded()

    try:
        # Validate box coordinates
        if request.x1 >= request.x2 or request.y1 >= request.y2:
            raise HTTPException(
                status_code=400,
                detail="Invalid bounding box: x1 must be < x2 and y1 must be < y2"
            )

        # Download image from URL
        image_data = await download_image_from_url(request.image_url)
        pil_image = validate_image(image_data)

        # Validate box is within image bounds
        w, h = pil_image.size
        if request.x1 < 0 or request.y1 < 0 or request.x2 > w or request.y2 > h:
            raise HTTPException(
                status_code=400,
                detail=f"Bounding box out of image bounds. Image size: {w}x{h}"
            )

        logger.info(f"Processing URL image with box: [{request.x1}, {request.y1}, {request.x2}, {request.y2}]")

        # Set image in processor
        inference_state = processor.set_image(pil_image)

        # Add box prompt
        output = processor.add_geometric_prompt(
            state=inference_state,
            boxes=[[request.x1, request.y1, request.x2, request.y2]]
        )

        # Extract results
        masks = output.get("masks")
        scores = output.get("scores")

        if masks is None or len(masks) == 0:
            return SegmentationResponse(
                success=True,
                num_masks=0,
                masks=[],
                message="No objects found in the specified box"
            )

        # Convert to numpy
        if hasattr(masks, 'cpu'):
            masks = masks.cpu().numpy()
        if hasattr(scores, 'cpu'):
            scores = scores.cpu().numpy()

        # Return visualization if requested
        if request.return_visualization:
            viz_image = create_visualization(pil_image, masks, [[request.x1, request.y1, request.x2, request.y2]])
            viz_base64 = base64.b64encode(viz_image).decode()
            return SegmentationResponse(
                success=True,
                num_masks=1,
                masks=[viz_base64],
                message="Visualization returned in masks[0]"
            )

        # Convert masks to base64
        masks_base64 = masks_to_base64(masks)

        return SegmentationResponse(
            success=True,
            num_masks=len(masks_base64),
            masks=masks_base64,
            boxes=[[request.x1, request.y1, request.x2, request.y2]],
            scores=scores.tolist() if scores is not None else None
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in box segmentation: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Segmentation failed: {str(e)}"
        )

@app.post("/segment/points/url", response_model=SegmentationResponse)
async def segment_with_points_url(request: PointSegmentURLRequest):
    """
    Segment objects using point prompts (URL input).

    This endpoint accepts an image URL and downloads it for processing.
    """
    check_model_loaded()

    try:
        # Validate same length
        if not (len(request.points_x) == len(request.points_y) == len(request.labels)):
            raise HTTPException(
                status_code=400,
                detail="Points X, Y, and labels must have the same length"
            )

        # Validate labels are 0 or 1
        if not all(l in [0, 1] for l in request.labels):
            raise HTTPException(
                status_code=400,
                detail="Labels must be 0 (negative) or 1 (positive)"
            )

        # Download image from URL
        image_data = await download_image_from_url(request.image_url)
        pil_image = validate_image(image_data)

        # Validate points are within image
        w, h = pil_image.size
        for x, y in zip(request.points_x, request.points_y):
            if x < 0 or y < 0 or x >= w or y >= h:
                raise HTTPException(
                    status_code=400,
                    detail=f"Point ({x}, {y}) is out of image bounds ({w}x{h})"
                )

        logger.info(f"Processing URL image with {len(request.points_x)} points")

        # Set image in processor
        inference_state = processor.set_image(pil_image)

        # Prepare points
        points = [[x, y] for x, y in zip(request.points_x, request.points_y)]

        # Add point prompts
        output = processor.add_geometric_prompt(
            state=inference_state,
            points=points,
            labels=request.labels
        )

        # Extract results
        masks = output.get("masks")
        scores = output.get("scores")

        if masks is None or len(masks) == 0:
            return SegmentationResponse(
                success=True,
                num_masks=0,
                masks=[],
                message="No objects found at the specified points"
            )

        # Convert to numpy
        if hasattr(masks, 'cpu'):
            masks = masks.cpu().numpy()
        if hasattr(scores, 'cpu'):
            scores = scores.cpu().numpy()

        # Return visualization if requested
        if request.return_visualization:
            viz_image = create_visualization(pil_image, masks)
            viz_base64 = base64.b64encode(viz_image).decode()
            return SegmentationResponse(
                success=True,
                num_masks=1,
                masks=[viz_base64],
                message="Visualization returned in masks[0]"
            )

        # Convert masks to base64
        masks_base64 = masks_to_base64(masks)

        return SegmentationResponse(
            success=True,
            num_masks=len(masks_base64),
            masks=masks_base64,
            scores=scores.tolist() if scores is not None else None
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in point segmentation: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Segmentation failed: {str(e)}"
        )

@app.post("/segment/auto/url", response_model=SegmentationResponse)
async def segment_automatic_url(request: AutoSegmentURLRequest):
    """
    Automatically segment all objects in an image (URL input).

    This endpoint accepts an image URL and downloads it for processing.
    """
    check_model_loaded()

    try:
        # Download image from URL
        image_data = await download_image_from_url(request.image_url)
        pil_image = validate_image(image_data)

        logger.info("Processing URL image with automatic segmentation")

        # Set image in processor
        inference_state = processor.set_image(pil_image)

        # Run automatic segmentation
        output = processor.generate_masks(
            state=inference_state,
            points_per_side=request.points_per_side,
            pred_iou_thresh=request.pred_iou_thresh,
            stability_score_thresh=request.stability_score_thresh
        )

        # Extract results
        masks = output.get("masks")
        scores = output.get("scores")

        if masks is None or len(masks) == 0:
            return SegmentationResponse(
                success=True,
                num_masks=0,
                masks=[],
                message="No objects found in automatic segmentation"
            )

        # Convert to numpy
        if hasattr(masks, 'cpu'):
            masks = masks.cpu().numpy()
        if hasattr(scores, 'cpu'):
            scores = scores.cpu().numpy()

        logger.info(f"Found {len(masks)} objects")

        # Return visualization if requested
        if request.return_visualization:
            viz_image = create_visualization(pil_image, masks)
            viz_base64 = base64.b64encode(viz_image).decode()
            return SegmentationResponse(
                success=True,
                num_masks=1,
                masks=[viz_base64],
                message="Visualization returned in masks[0]"
            )

        # Convert masks to base64
        masks_base64 = masks_to_base64(masks)

        return SegmentationResponse(
            success=True,
            num_masks=len(masks_base64),
            masks=masks_base64,
            scores=scores.tolist() if scores is not None else None
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in automatic segmentation: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Segmentation failed: {str(e)}"
        )

# WebSocket endpoint for real-time segmentation
@app.websocket("/ws/realtime")
async def websocket_realtime_segmentation(websocket: WebSocket):
    """
    WebSocket endpoint for real-time click/touch-based segmentation.

    Protocol:
    Client sends JSON messages:
    - {"type": "init", "image": "base64_encoded_image"} - Initialize session with image
    - {"type": "click", "x": 100, "y": 200, "label": 1} - Add click point (label: 1=foreground, 0=background)
    - {"type": "segment"} - Trigger segmentation with current points
    - {"type": "reset"} - Clear all points
    - {"type": "close"} - End session

    Server responds with JSON:
    - {"type": "session_created", "session_id": "uuid"}
    - {"type": "image_loaded", "width": 1920, "height": 1080}
    - {"type": "point_added", "x": 100, "y": 200, "label": 1}
    - {"type": "segmentation_result", "mask": "base64_png", "score": 0.95}
    - {"type": "error", "message": "error description"}
    """
    await websocket.accept()
    session_id = str(uuid.uuid4())
    session_data = {
        "inference_state": None,
        "image": None,
        "points_x": [],
        "points_y": [],
        "labels": [],
        "last_activity": datetime.now()
    }
    realtime_sessions[session_id] = session_data

    try:
        # Send session ID to client
        await websocket.send_json({
            "type": "session_created",
            "session_id": session_id
        })

        while True:
            # Receive message from client
            data = await websocket.receive_json()
            msg_type = data.get("type")

            # Update last activity
            session_data["last_activity"] = datetime.now()

            if msg_type == "init":
                # Initialize session with image
                try:
                    if not model_loaded:
                        await websocket.send_json({
                            "type": "error",
                            "message": "Model not loaded yet. Please wait."
                        })
                        continue

                    image_base64 = data.get("image")
                    if not image_base64:
                        await websocket.send_json({
                            "type": "error",
                            "message": "No image provided"
                        })
                        continue

                    # Decode image
                    if ',' in image_base64:
                        image_base64 = image_base64.split(',', 1)[1]

                    image_bytes = base64.b64decode(image_base64)
                    pil_image = validate_image(image_bytes)

                    # Set image in processor
                    inference_state = processor.set_image(pil_image)

                    # Store in session
                    session_data["inference_state"] = inference_state
                    session_data["image"] = pil_image
                    session_data["points_x"] = []
                    session_data["points_y"] = []
                    session_data["labels"] = []

                    await websocket.send_json({
                        "type": "image_loaded",
                        "width": pil_image.width,
                        "height": pil_image.height
                    })

                except Exception as e:
                    logger.error(f"Error loading image: {e}")
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Failed to load image: {str(e)}"
                    })

            elif msg_type == "click":
                # Add click point
                try:
                    x = data.get("x")
                    y = data.get("y")
                    label = data.get("label", 1)  # Default to foreground

                    if x is None or y is None:
                        await websocket.send_json({
                            "type": "error",
                            "message": "Invalid click coordinates"
                        })
                        continue

                    # Add point to session
                    session_data["points_x"].append(int(x))
                    session_data["points_y"].append(int(y))
                    session_data["labels"].append(int(label))

                    await websocket.send_json({
                        "type": "point_added",
                        "x": int(x),
                        "y": int(y),
                        "label": int(label),
                        "total_points": len(session_data["points_x"])
                    })

                    # Automatically trigger segmentation
                    if session_data["image"] is not None and len(session_data["points_x"]) > 0:
                        try:
                            # Use SAM3 Tracker for point-based prompts (Transformers API)
                            # Convert points to the required format: [batch][objects][points_per_object][coordinates]
                            points = [[x, y] for x, y in zip(session_data["points_x"], session_data["points_y"])]
                            input_points = [[points]]  # 4D: [1 image][1 object][list of [x,y] points]
                            input_labels = [[session_data["labels"]]]  # 3D: [1 image][1 object][list of labels]

                            logger.info(f"Processing {len(points)} points with tracker model")

                            # Process with SAM3 Tracker
                            import torch
                            inputs = tracker_processor(
                                images=session_data["image"],
                                input_points=input_points,
                                input_labels=input_labels,
                                return_tensors="pt"
                            ).to(tracker_model.device)

                            with torch.no_grad():
                                outputs = tracker_model(**inputs, multimask_output=False)

                            # Post-process masks to original image size
                            masks = tracker_processor.post_process_masks(
                                outputs.pred_masks.cpu(),
                                inputs["original_sizes"]
                            )[0]  # Get masks for first (only) image

                            # Extract scores
                            scores = outputs.iou_scores.cpu().numpy() if hasattr(outputs, 'iou_scores') else None

                            # Convert masks to numpy
                            if hasattr(masks, 'numpy'):
                                masks = masks.numpy()
                            elif isinstance(masks, list):
                                masks = np.array(masks)

                            logger.info(f"Masks shape after conversion: {masks.shape}, dtype: {masks.dtype}")

                            # Take the first (best) mask
                            # masks shape: [num_objects, num_predictions_per_object, H, W]
                            if len(masks.shape) == 4:
                                mask = masks[0, 0]  # First object, first prediction
                            elif len(masks.shape) == 3:
                                mask = masks[0]  # First mask
                            elif len(masks.shape) == 2:
                                mask = masks
                            else:
                                logger.error(f"Unexpected mask shape: {masks.shape}")
                                raise ValueError(f"Unexpected mask shape: {masks.shape}")

                            # Ensure mask is numpy array
                            if hasattr(mask, 'numpy'):
                                mask = mask.numpy()
                            elif not isinstance(mask, np.ndarray):
                                mask = np.array(mask)

                            logger.info(f"Final mask shape: {mask.shape}, dtype: {mask.dtype}, type: {type(mask)}")

                            # Extract score properly to avoid deprecation warning
                            if scores is not None and scores.size > 0:
                                score = float(scores.flat[0])  # Use flat indexing to get scalar
                            else:
                                score = 0.0

                            # Convert mask to base64 PNG
                            mask_base64 = masks_to_base64([mask])[0]

                            await websocket.send_json({
                                "type": "segmentation_result",
                                "mask": mask_base64,
                                "score": score,
                                "num_points": len(session_data["points_x"])
                            })

                        except Exception as e:
                            logger.error(f"Error in segmentation: {e}")
                            logger.error(traceback.format_exc())
                            await websocket.send_json({
                                "type": "error",
                                "message": f"Segmentation failed: {str(e)}"
                            })

                except Exception as e:
                    logger.error(f"Error adding point: {e}")
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Failed to add point: {str(e)}"
                    })

            elif msg_type == "reset":
                # Clear all points
                session_data["points_x"] = []
                session_data["points_y"] = []
                session_data["labels"] = []

                await websocket.send_json({
                    "type": "points_cleared"
                })

            elif msg_type == "close":
                # End session
                break

            else:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Unknown message type: {msg_type}"
                })

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        logger.error(traceback.format_exc())
    finally:
        # Clean up session
        if session_id in realtime_sessions:
            del realtime_sessions[session_id]
        logger.info(f"Session {session_id} cleaned up")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
