from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from PIL import Image
import torch
from transformers import Sam3Processor, Sam3Model
import io
import base64
import numpy as np
from typing import Optional, List

app = FastAPI(title="SAM3 Segmentation API")

# Global variables for model and processor
model = None
processor = None
device = None

@app.on_event("startup")
async def load_model():
    """Load the SAM3 model on startup"""
    global model, processor, device

    print("Loading SAM3 model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = Sam3Model.from_pretrained("facebook/sam3").to(device)
    processor = Sam3Processor.from_pretrained("facebook/sam3")

    print("Model loaded successfully!")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "model": "SAM3",
        "device": device
    }

@app.post("/segment/text")
async def segment_with_text(
    image: UploadFile = File(...),
    text: str = Form(...),
    threshold: float = Form(0.5),
    mask_threshold: float = Form(0.5)
):
    """
    Segment objects in an image using a text prompt

    Args:
        image: The image file to segment
        text: Text description of objects to segment (e.g., "ear", "person", "car")
        threshold: Confidence threshold for object detection (default: 0.5)
        mask_threshold: Threshold for mask generation (default: 0.5)

    Returns:
        JSON with number of objects found and their masks (base64 encoded)
    """
    try:
        # Read and process image
        image_data = await image.read()
        pil_image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Prepare inputs
        inputs = processor(images=pil_image, text=text, return_tensors="pt").to(device)

        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)

        # Post-process results
        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=threshold,
            mask_threshold=mask_threshold,
            target_sizes=inputs.get("original_sizes").tolist()
        )[0]

        # Convert masks to base64 for transmission
        masks_base64 = []
        for mask in results['masks']:
            mask_array = mask.cpu().numpy().astype(np.uint8) * 255
            mask_image = Image.fromarray(mask_array)
            buffer = io.BytesIO()
            mask_image.save(buffer, format="PNG")
            mask_b64 = base64.b64encode(buffer.getvalue()).decode()
            masks_base64.append(mask_b64)

        return JSONResponse({
            "success": True,
            "num_objects": len(results['masks']),
            "scores": results['scores'].cpu().tolist() if 'scores' in results else [],
            "masks": masks_base64,
            "prompt": text
        })

    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

@app.post("/segment/box")
async def segment_with_box(
    image: UploadFile = File(...),
    x1: int = Form(...),
    y1: int = Form(...),
    x2: int = Form(...),
    y2: int = Form(...)
):
    """
    Segment objects in an image using a bounding box prompt

    Args:
        image: The image file to segment
        x1, y1: Top-left corner of bounding box
        x2, y2: Bottom-right corner of bounding box

    Returns:
        JSON with segmentation mask (base64 encoded)
    """
    try:
        # Read and process image
        image_data = await image.read()
        pil_image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Prepare bounding box
        box_xyxy = [x1, y1, x2, y2]
        input_boxes = [[box_xyxy]]
        input_boxes_labels = [[1]]  # 1 = positive box

        # Prepare inputs
        inputs = processor(
            images=pil_image,
            input_boxes=input_boxes,
            input_boxes_labels=input_boxes_labels,
            return_tensors="pt"
        ).to(device)

        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)

        # Post-process results
        results = processor.post_process_instance_segmentation(
            outputs,
            target_sizes=inputs.get("original_sizes").tolist()
        )[0]

        # Convert masks to base64
        masks_base64 = []
        for mask in results['masks']:
            mask_array = mask.cpu().numpy().astype(np.uint8) * 255
            mask_image = Image.fromarray(mask_array)
            buffer = io.BytesIO()
            mask_image.save(buffer, format="PNG")
            mask_b64 = base64.b64encode(buffer.getvalue()).decode()
            masks_base64.append(mask_b64)

        return JSONResponse({
            "success": True,
            "num_masks": len(results['masks']),
            "masks": masks_base64,
            "bounding_box": box_xyxy
        })

    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

@app.post("/segment/points")
async def segment_with_points(
    image: UploadFile = File(...),
    points_x: str = Form(...),  # Comma-separated x coordinates
    points_y: str = Form(...),  # Comma-separated y coordinates
    labels: str = Form(...)      # Comma-separated labels (1 for positive, 0 for negative)
):
    """
    Segment objects in an image using point prompts

    Args:
        image: The image file to segment
        points_x: Comma-separated x coordinates (e.g., "100,200,300")
        points_y: Comma-separated y coordinates (e.g., "150,250,350")
        labels: Comma-separated labels, 1 for positive point, 0 for negative (e.g., "1,1,0")

    Returns:
        JSON with segmentation mask (base64 encoded)
    """
    try:
        # Read and process image
        image_data = await image.read()
        pil_image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Parse points
        x_coords = [int(x.strip()) for x in points_x.split(",")]
        y_coords = [int(y.strip()) for y in points_y.split(",")]
        point_labels = [int(l.strip()) for l in labels.split(",")]

        input_points = [[[x, y] for x, y in zip(x_coords, y_coords)]]

        # Prepare inputs
        inputs = processor(
            images=pil_image,
            input_points=input_points,
            input_labels=[point_labels],
            return_tensors="pt"
        ).to(device)

        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)

        # Post-process results
        results = processor.post_process_instance_segmentation(
            outputs,
            target_sizes=inputs.get("original_sizes").tolist()
        )[0]

        # Convert masks to base64
        masks_base64 = []
        for mask in results['masks']:
            mask_array = mask.cpu().numpy().astype(np.uint8) * 255
            mask_image = Image.fromarray(mask_array)
            buffer = io.BytesIO()
            mask_image.save(buffer, format="PNG")
            mask_b64 = base64.b64encode(buffer.getvalue()).decode()
            masks_base64.append(mask_b64)

        return JSONResponse({
            "success": True,
            "num_masks": len(results['masks']),
            "masks": masks_base64,
            "points": list(zip(x_coords, y_coords)),
            "point_labels": point_labels
        })

    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
