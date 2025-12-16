from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
import io
import base64
import numpy as np
import logging
import os
import uuid
from datetime import datetime
import torch
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SAM3 Real-time Segmentation",
    description="WebSocket-based real-time segmentation with SAM3",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    logger.info(f"Static files mounted from {static_dir}")

model = None
processor = None
device = None
tracker_model = None
tracker_processor = None
realtime_sessions = {}

MAX_SESSIONS = 100
SESSION_TIMEOUT = 3600
MAX_IMAGE_SIZE = 10 * 1024 * 1024
MAX_OBJECTS_PER_SESSION = 50
MAX_POINTS_PER_OBJECT = 100
PROXIMITY_THRESHOLD = 100

@app.on_event("startup")
async def load_model():
    global model, processor, device, tracker_model, tracker_processor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not hf_token:
        logger.warning("No Hugging Face token found in environment (HF_TOKEN or HUGGING_FACE_HUB_TOKEN)")
        logger.warning("You may need to set HF_TOKEN env variable if model download fails")

    from transformers import Sam3TrackerProcessor, Sam3TrackerModel

    try:
        logger.info("Loading SAM3 Tracker model (required for WebSocket)...")
        tracker_model = Sam3TrackerModel.from_pretrained(
            "facebook/sam3",
            token=hf_token
        ).to(device)
        tracker_processor = Sam3TrackerProcessor.from_pretrained(
            "facebook/sam3",
            token=hf_token
        )
        tracker_model.eval()
        logger.info("SAM3 Tracker loaded successfully")
    except Exception as e:
        logger.error(f"Tracker model loading failed: {e}")
        logger.error("Make sure you have:")
        logger.error("1. Requested access at: https://huggingface.co/facebook/sam3")
        logger.error("2. Set HF_TOKEN environment variable with your Hugging Face token")
        raise

    try:
        logger.info("Loading SAM3 native model (optional)...")
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        model = build_sam3_image_model()
        if device == "cuda":
            model = model.cuda()
        processor = Sam3Processor(model)
        logger.info("SAM3 native model loaded successfully")
    except Exception as e:
        logger.warning(f"Native model loading failed (non-critical): {e}")
        logger.info("Continuing with Tracker model only (sufficient for WebSocket)")

def decode_base64_image(image_str: str) -> bytes:
    if image_str.startswith('data:image'):
        image_str = image_str.split(',', 1)[1]
    return base64.b64decode(image_str)

def validate_image(image_bytes: bytes) -> Image.Image:
    if len(image_bytes) > MAX_IMAGE_SIZE:
        raise ValueError(f"Image too large: {len(image_bytes)} bytes (max {MAX_IMAGE_SIZE})")

    image = Image.open(io.BytesIO(image_bytes))
    if image.mode != "RGB":
        image = image.convert("RGB")

    if max(image.size) > 2048:
        ratio = 2048 / max(image.size)
        new_size = tuple(int(dim * ratio) for dim in image.size)
        image = image.resize(new_size, Image.LANCZOS)

    if min(image.size) < 32:
        raise ValueError(f"Image too small: {image.size} (min 32px)")

    return image

def masks_to_base64(masks: np.ndarray) -> list:
    result = []
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks.squeeze(1)

    for i in range(masks.shape[0]):
        mask = masks[i]
        if mask.dtype == bool:
            mask_uint8 = (mask * 255).astype(np.uint8)
        else:
            mask_uint8 = ((mask > 0.5) * 255).astype(np.uint8)

        mask_image = Image.fromarray(mask_uint8, mode='L')
        buffer = io.BytesIO()
        mask_image.save(buffer, format='PNG')
        result.append(base64.b64encode(buffer.getvalue()).decode('utf-8'))

    return result

def cleanup_old_sessions():
    current_time = datetime.now()
    expired = [
        sid for sid, data in realtime_sessions.items()
        if (current_time - data.get('last_activity', current_time)).total_seconds() > SESSION_TIMEOUT
    ]
    for sid in expired:
        logger.info(f"Removing expired session: {sid}")
        del realtime_sessions[sid]

def enforce_session_limit():
    if len(realtime_sessions) > MAX_SESSIONS:
        sorted_sessions = sorted(
            realtime_sessions.items(),
            key=lambda x: x[1].get('last_activity', datetime.min)
        )
        num_to_remove = len(realtime_sessions) - MAX_SESSIONS
        for i in range(num_to_remove):
            sid = sorted_sessions[i][0]
            logger.info(f"Removing oldest session: {sid}")
            del realtime_sessions[sid]

@app.get("/")
async def root():
    return {
        "status": "ok",
        "service": "SAM3 Real-time Segmentation",
        "websocket_endpoint": "/ws/realtime"
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": tracker_model is not None,
        "device": str(device),
        "active_sessions": len(realtime_sessions)
    }

@app.get("/app", response_class=HTMLResponse)
async def serve_app():
    html_path = os.path.join(static_dir, "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r") as f:
            return HTMLResponse(content=f.read())
    raise HTTPException(status_code=404, detail="Web app not found")

@app.websocket("/ws/realtime")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid.uuid4())
    session_data = {
        "image": None,
        "objects": [],
        "last_activity": datetime.now()
    }

    cleanup_old_sessions()
    enforce_session_limit()
    realtime_sessions[session_id] = session_data

    try:
        await websocket.send_json({"type": "session_created", "session_id": session_id})

        while True:
            try:
                data = await websocket.receive_json()
            except Exception as e:
                logger.error(f"Failed to parse JSON: {e}")
                await websocket.send_json({"type": "error", "message": "Invalid JSON"})
                continue

            msg_type = data.get("type")
            session_data["last_activity"] = datetime.now()

            if msg_type == "init":
                if tracker_model is None:
                    await websocket.send_json({"type": "error", "message": "Model not loaded"})
                    continue

                try:
                    image_bytes = decode_base64_image(data.get("image", ""))
                    image = validate_image(image_bytes)
                    session_data["image"] = image
                    session_data["objects"] = []

                    await websocket.send_json({
                        "type": "image_loaded",
                        "width": image.width,
                        "height": image.height
                    })
                except Exception as e:
                    logger.error(f"Image loading error: {e}")
                    await websocket.send_json({"type": "error", "message": str(e)})

            elif msg_type == "click":
                image = session_data.get("image")
                if image is None:
                    await websocket.send_json({"type": "error", "message": "No image loaded"})
                    continue

                try:
                    x = data.get("x")
                    y = data.get("y")
                    label = data.get("label", 1)

                    if x is None or y is None:
                        await websocket.send_json({"type": "error", "message": "Missing x or y"})
                        continue

                    x, y = int(x), int(y)

                    if not (0 <= x < image.width and 0 <= y < image.height):
                        await websocket.send_json({
                            "type": "error",
                            "message": f"Point ({x}, {y}) outside image bounds ({image.width}x{image.height})"
                        })
                        continue

                    if label not in [0, 1]:
                        label = 1

                    assigned_to = None
                    min_distance = float('inf')

                    for idx, obj in enumerate(session_data["objects"]):
                        for px, py in obj["points"]:
                            dist = np.sqrt((x - px)**2 + (y - py)**2)
                            if dist < min_distance:
                                min_distance = dist
                                assigned_to = idx

                    if min_distance <= PROXIMITY_THRESHOLD and assigned_to is not None:
                        obj = session_data["objects"][assigned_to]
                        if len(obj["points"]) >= MAX_POINTS_PER_OBJECT:
                            await websocket.send_json({
                                "type": "error",
                                "message": f"Max points per object ({MAX_POINTS_PER_OBJECT}) reached"
                            })
                            continue
                        obj["points"].append([x, y])
                        obj["labels"].append(label)
                    else:
                        if len(session_data["objects"]) >= MAX_OBJECTS_PER_SESSION:
                            await websocket.send_json({
                                "type": "error",
                                "message": f"Max objects ({MAX_OBJECTS_PER_SESSION}) reached"
                            })
                            continue
                        session_data["objects"].append({
                            "points": [[x, y]],
                            "labels": [label]
                        })

                    await websocket.send_json({
                        "type": "point_added",
                        "x": x,
                        "y": y,
                        "label": label
                    })

                    if tracker_model is not None:
                        try:
                            num_objects = len(session_data["objects"])
                            objects_points = [obj["points"] for obj in session_data["objects"]]
                            objects_labels = [obj["labels"] for obj in session_data["objects"]]

                            all_masks = []
                            for obj_idx in range(num_objects):
                                points = objects_points[obj_idx]
                                labels = objects_labels[obj_idx]

                                inputs = tracker_processor(
                                    image,
                                    input_points=[[points]],
                                    input_labels=[[labels]],
                                    return_tensors="pt"
                                ).to(tracker_model.device)

                                with torch.no_grad():
                                    outputs = tracker_model(**inputs, multimask_output=False)

                                pred_masks = outputs.pred_masks.cpu()

                                if hasattr(tracker_processor, 'post_process_masks'):
                                    original_size = inputs.get("original_sizes", torch.tensor([[image.height, image.width]]))
                                    reshaped_size = inputs.get("reshaped_input_sizes", original_size)
                                    masks = tracker_processor.post_process_masks(
                                        pred_masks,
                                        original_size.cpu(),
                                        reshaped_size.cpu()
                                    )[0]
                                else:
                                    masks = pred_masks

                                all_masks.append(masks)

                            first_mask_shape = all_masks[0].shape
                            if len(first_mask_shape) >= 3:
                                composite_mask = np.zeros(first_mask_shape[-2:], dtype=bool)
                                for masks in all_masks:
                                    mask = masks.squeeze().numpy()
                                    if mask.ndim > 2:
                                        mask = mask[0]
                                    composite_mask = composite_mask | (mask > 0.5)
                            else:
                                composite_mask = all_masks[0].squeeze().numpy() > 0.5

                            mask_b64 = masks_to_base64(np.array([composite_mask]))[0]

                            await websocket.send_json({
                                "type": "segmentation_result",
                                "mask": mask_b64,
                                "num_objects": num_objects
                            })
                        except Exception as e:
                            logger.error(f"Segmentation error: {e}")
                            import traceback
                            logger.error(traceback.format_exc())
                            await websocket.send_json({"type": "error", "message": f"Segmentation failed: {str(e)}"})

                except Exception as e:
                    logger.error(f"Click processing error: {e}")
                    await websocket.send_json({"type": "error", "message": str(e)})

            elif msg_type == "reset":
                session_data["objects"] = []
                await websocket.send_json({"type": "reset_confirmed"})

            elif msg_type == "close":
                break

            else:
                await websocket.send_json({"type": "error", "message": f"Unknown message type: {msg_type}"})

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        if session_id in realtime_sessions:
            del realtime_sessions[session_id]
        logger.info(f"Session cleaned up: {session_id}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
