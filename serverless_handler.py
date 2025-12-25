import runpod
import base64
import io
import os
import torch
import numpy as np
from PIL import Image
from dotenv import load_dotenv
import cv2

load_dotenv()

tracker_model = None
tracker_processor = None
matting_model = None
matting_processor = None
device = None

def load_models():
    global tracker_model, tracker_processor, device

    if tracker_model is not None:
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")

    from transformers import Sam3TrackerProcessor, Sam3TrackerModel

    print("Loading SAM3 Tracker model...")
    tracker_model = Sam3TrackerModel.from_pretrained(
        "facebook/sam3",
        token=hf_token
    ).to(device)
    tracker_processor = Sam3TrackerProcessor.from_pretrained(
        "facebook/sam3",
        token=hf_token
    )
    tracker_model.eval()
    print("Model loaded successfully")

def load_matting_model():
    global matting_model, matting_processor, device

    if matting_model is not None:
        return

    from transformers import VitMatteForImageMatting, VitMatteImageProcessor

    print("Loading ViTMatte model...")
    matting_processor = VitMatteImageProcessor.from_pretrained("hustvl/vitmatte-small-composition-1k")
    matting_model = VitMatteForImageMatting.from_pretrained("hustvl/vitmatte-small-composition-1k").to(device)
    matting_model.eval()
    print("ViTMatte loaded successfully")

def decode_base64_image(image_str: str) -> Image.Image:
    if image_str.startswith('data:image'):
        image_str = image_str.split(',', 1)[1]
    image_bytes = base64.b64decode(image_str)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")

def mask_to_base64(mask: np.ndarray) -> str:
    if mask.dtype == bool:
        mask_uint8 = (mask * 255).astype(np.uint8)
    else:
        mask_uint8 = ((mask > 0.5) * 255).astype(np.uint8)

    mask_image = Image.fromarray(mask_uint8, mode='L')
    buffer = io.BytesIO()
    mask_image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def generate_trimap(mask: np.ndarray, erode_size: int = 10, dilate_size: int = 10) -> np.ndarray:
    mask_uint8 = (mask * 255).astype(np.uint8) if mask.dtype == bool else mask.astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(mask_uint8, kernel, iterations=erode_size)
    dilated = cv2.dilate(mask_uint8, kernel, iterations=dilate_size)

    trimap = np.zeros_like(mask_uint8)
    trimap[dilated > 127] = 128
    trimap[eroded > 127] = 255

    return trimap

def apply_matting(image: Image.Image, mask: np.ndarray) -> np.ndarray:
    load_matting_model()

    trimap = generate_trimap(mask)
    trimap_image = Image.fromarray(trimap, mode='L')

    inputs = matting_processor(images=image, trimaps=trimap_image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = matting_model(**inputs)

    alpha = outputs.alphas[0, 0].cpu().numpy()
    alpha = (alpha * 255).astype(np.uint8)

    return alpha

def extract_layer(image: Image.Image, mask: np.ndarray) -> bytes:
    img_array = np.array(image)
    h, w = img_array.shape[:2]

    if mask.dtype == bool:
        alpha = (mask * 255).astype(np.uint8)
    elif mask.max() <= 1.0:
        alpha = (mask * 255).astype(np.uint8)
    else:
        alpha = mask.astype(np.uint8)

    if alpha.shape != (h, w):
        alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)

    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, :3] = img_array
    rgba[:, :, 3] = alpha

    rgba_image = Image.fromarray(rgba, mode='RGBA')
    buffer = io.BytesIO()
    rgba_image.save(buffer, format='PNG')
    return buffer.getvalue()

def segment_image(image: Image.Image, points: list, labels: list) -> np.ndarray:
    inputs = tracker_processor(
        image,
        input_points=[[points]],
        input_labels=[[labels]],
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = tracker_model(**inputs, multimask_output=False)

    masks = tracker_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"]
    )[0]

    mask = masks.squeeze().numpy()
    return mask

def handler(job):
    """
    RunPod serverless handler for SAM3 segmentation and layer extraction.

    Actions:
    - "segment" (default): Returns binary mask
    - "extract_layer": Returns RGBA PNG with transparent background

    Input format:
    {
        "input": {
            "action": "segment" | "extract_layer",
            "image": "base64_string",
            "points": [[x1, y1], [x2, y2], ...],
            "labels": [1, 1, 0, ...],
            "use_matting": false  // Optional, for extract_layer
        }
    }

    Output format (segment):
    {
        "mask": "base64_png_string",
        "mask_shape": [height, width]
    }

    Output format (extract_layer):
    {
        "layer": "base64_rgba_png",
        "width": 1920,
        "height": 1080
    }
    """
    try:
        load_models()

        job_input = job.get("input", {})
        action = job_input.get("action", "segment")

        image_b64 = job_input.get("image")
        points = job_input.get("points", [])
        labels = job_input.get("labels", [])

        if not image_b64:
            return {"error": "No image provided"}

        if not points:
            return {"error": "No points provided"}

        if len(points) != len(labels):
            labels = [1] * len(points)

        image = decode_base64_image(image_b64)
        mask = segment_image(image, points, labels)
        mask_binary = mask > 0.5

        if action == "segment":
            mask_b64 = mask_to_base64(mask_binary)
            return {
                "mask": mask_b64,
                "mask_shape": list(mask_binary.shape),
                "num_points": len(points)
            }

        elif action == "extract_layer":
            use_matting = job_input.get("use_matting", False)

            if use_matting:
                alpha = apply_matting(image, mask_binary)
            else:
                alpha = (mask_binary * 255).astype(np.uint8)

            layer_bytes = extract_layer(image, alpha)
            layer_b64 = base64.b64encode(layer_bytes).decode('utf-8')

            return {
                "layer": layer_b64,
                "width": image.width,
                "height": image.height,
                "use_matting": use_matting
            }

        else:
            return {"error": f"Unknown action: {action}. Use 'segment' or 'extract_layer'"}

    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
