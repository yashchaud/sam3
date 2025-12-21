import runpod
import base64
import io
import os
import torch
import numpy as np
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

tracker_model = None
tracker_processor = None
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

def handler(job):
    """
    RunPod serverless handler for SAM3 segmentation.

    Input format:
    {
        "input": {
            "image": "base64_string",
            "points": [[x1, y1], [x2, y2], ...],
            "labels": [1, 1, 0, ...]  # 1=foreground, 0=background
        }
    }

    Output format:
    {
        "mask": "base64_png_string",
        "mask_shape": [height, width]
    }
    """
    try:
        load_models()

        job_input = job.get("input", {})

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
        mask_binary = mask > 0.5

        mask_b64 = mask_to_base64(mask_binary)

        return {
            "mask": mask_b64,
            "mask_shape": list(mask_binary.shape),
            "num_points": len(points)
        }

    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
