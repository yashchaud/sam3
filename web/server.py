"""FastAPI server for anomaly detection web interface.

Pipeline: VLM (OpenRouter) -> SAM3 Segmentation
Configuration is loaded from environment variables.
Models are auto-loaded on server startup.
"""

import asyncio
import base64
import io
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import our modules
from anomaly_detection.config import get_config, EnvironmentConfig
from anomaly_detection.realtime import RealtimeVideoProcessor, RealtimeConfig, FrameSource
from anomaly_detection.vlm import VLMConfig, VLMProvider, GridConfig
from anomaly_detection.utils import load_image, draw_detections, draw_mask_overlay

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager - auto-load models on startup."""
    # Startup: Load models automatically
    logger.info("Server starting - loading models from environment configuration...")

    try:
        env_config = get_config()
        env_config.print_config()

        # Validate configuration
        errors = env_config.validate()
        if errors:
            logger.error(f"Configuration errors: {'; '.join(errors)}")
            logger.warning("Models NOT loaded due to configuration errors. Fix .env and restart.")
        else:
            # Build VLM config
            vlm_config = VLMConfig(
                provider=VLMProvider.OPENROUTER,
                openrouter_api_key=env_config.openrouter_api_key,
                openrouter_model=env_config.openrouter_model,
                process_every_n_frames=env_config.vlm_every_n_frames,
                max_generation_frames=env_config.vlm_max_generation_frames,
                grid_config=GridConfig(cols=3, rows=3),
            )

            # Build main config
            config = RealtimeConfig(
                hf_token=env_config.hf_token,
                vlm_config=vlm_config,
                confidence_threshold=env_config.confidence_threshold,
                device=env_config.device,
            )

            # Create and load processor with debug output enabled
            global state
            state.config = config
            state.processor = RealtimeVideoProcessor(config, debug_output_dir="output/debug")
            state.processor.load()

            logger.info("Models loaded successfully!")
            logger.info(f"  SAM3: facebook/sam3 (HuggingFace)")
            logger.info(f"  VLM: {env_config.openrouter_model}")

    except Exception as e:
        logger.error(f"Failed to auto-load models: {e}")
        logger.warning("Server running but models NOT loaded. Use /api/load to retry.")

    yield  # Server runs here

    # Shutdown: Cleanup
    logger.info("Server shutting down - unloading models...")
    if state.processor is not None:
        state.processor.unload()
        state.processor = None
    logger.info("Cleanup complete.")


app = FastAPI(title="Anomaly Detection", version="0.3.0", lifespan=lifespan)

# CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
processor: Optional[RealtimeVideoProcessor] = None
processing_task: Optional[asyncio.Task] = None
active_connections: list[WebSocket] = []


# ============== Models ==============

class ProcessingStatus(BaseModel):
    is_running: bool
    frames_processed: int
    total_detections: int
    current_fps: float
    vlm_stats: dict


# ============== State Management ==============

@dataclass
class AppState:
    config: Optional[RealtimeConfig] = None
    processor: Optional[RealtimeVideoProcessor] = None
    is_processing: bool = False
    current_video_path: Optional[str] = None
    frames_processed: int = 0
    total_detections: int = 0
    results_buffer: list = field(default_factory=list)


state = AppState()


# ============== WebSocket Management ==============

async def broadcast_message(message: dict):
    """Send message to all connected WebSocket clients."""
    disconnected = []
    for ws in active_connections:
        try:
            await ws.send_json(message)
        except Exception:
            disconnected.append(ws)

    for ws in disconnected:
        active_connections.remove(ws)


# ============== API Endpoints ==============

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main web interface."""
    html_path = Path(__file__).parent / "static" / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text())
    return HTMLResponse(content="<h1>Static files not found. Run from correct directory.</h1>")


@app.get("/api/config")
async def get_current_config():
    """Get current configuration from environment."""
    env_config = get_config()
    models_loaded = state.processor is not None and state.processor.is_loaded()
    return {
        "sam3_model": "facebook/sam3 (HuggingFace)",
        "hf_token_set": bool(env_config.hf_token),
        "openrouter_model": env_config.openrouter_model,
        "openrouter_api_key_set": bool(env_config.openrouter_api_key),
        "vlm_every_n_frames": env_config.vlm_every_n_frames,
        "confidence_threshold": env_config.confidence_threshold,
        "device": env_config.device,
        "models_loaded": models_loaded,
    }


@app.post("/api/load")
async def load_models():
    """Load all models into memory using environment configuration."""
    global state

    try:
        # Get configuration from environment
        env_config = get_config()

        # Validate configuration
        errors = env_config.validate()
        if errors:
            raise HTTPException(status_code=400, detail="; ".join(errors))

        # Build VLM config from environment
        vlm_config = VLMConfig(
            provider=VLMProvider.OPENROUTER,
            openrouter_api_key=env_config.openrouter_api_key,
            openrouter_model=env_config.openrouter_model,
            process_every_n_frames=env_config.vlm_every_n_frames,
            max_generation_frames=env_config.vlm_max_generation_frames,
            grid_config=GridConfig(cols=3, rows=3),
        )

        # Build main config (VLM + SAM3 pipeline)
        state.config = RealtimeConfig(
            hf_token=env_config.hf_token,
            vlm_config=vlm_config,
            confidence_threshold=env_config.confidence_threshold,
            device=env_config.device,
        )

        if state.processor is not None:
            state.processor.unload()

        state.processor = RealtimeVideoProcessor(state.config, debug_output_dir="output/debug")
        state.processor.load()

        logger.info(f"Models loaded - SAM3: facebook/sam3 (HuggingFace), VLM: {env_config.openrouter_model}")

        return {
            "status": "ok",
            "message": "Models loaded",
            "config": {
                "sam3_model": "facebook/sam3 (HuggingFace)",
                "vlm_model": env_config.openrouter_model,
                "device": env_config.device,
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/unload")
async def unload_models():
    """Unload models from memory."""
    global state

    if state.processor is not None:
        state.processor.unload()
        state.processor = None

    return {"status": "ok", "message": "Models unloaded"}


@app.get("/api/status")
async def get_status():
    """Get current processing status."""
    global state

    vlm_stats = {}
    if state.processor and hasattr(state.processor, '_vlm_judge') and state.processor._vlm_judge:
        vlm_stats = state.processor._vlm_judge.get_stats()

    return ProcessingStatus(
        is_running=state.is_processing,
        frames_processed=state.frames_processed,
        total_detections=state.total_detections,
        current_fps=state.processor.get_stats().avg_fps if state.processor else 0.0,
        vlm_stats=vlm_stats,
    )


def annotate_image(image: np.ndarray, result) -> np.ndarray:
    """Draw annotations (masks, boxes, labels) on image."""
    annotated = image.copy()

    # Draw masks
    for anomaly in result.all_anomalies:
        if anomaly.mask:
            annotated = draw_mask_overlay(annotated, anomaly.mask.data, color=(255, 0, 0), alpha=0.4)

    # Draw boxes for anomalies
    for anomaly in result.all_anomalies:
        cv2.rectangle(
            annotated,
            (anomaly.bbox.x_min, anomaly.bbox.y_min),
            (anomaly.bbox.x_max, anomaly.bbox.y_max),
            (255, 0, 0),
            2,
        )
        label = f"{anomaly.defect_type}: {anomaly.detection_confidence:.0%}"
        cv2.putText(annotated, label, (anomaly.bbox.x_min, anomaly.bbox.y_min - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Draw structures
    for struct in result.structures:
        cv2.rectangle(
            annotated,
            (struct.bbox.x_min, struct.bbox.y_min),
            (struct.bbox.x_max, struct.bbox.y_max),
            (0, 255, 0),
            2,
        )

    return annotated


def encode_image_base64(image: np.ndarray) -> str:
    """Encode image to base64 JPEG string."""
    annotated_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.jpg', annotated_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buffer).decode('utf-8')


@app.post("/api/process/image")
async def process_image(file: UploadFile = File(...)):
    """Process a single uploaded image with incremental updates.

    Sends WebSocket updates after each processing step (global, each tile).
    Final HTTP response contains complete results.
    """
    global state

    if state.processor is None or not state.processor.is_loaded():
        raise HTTPException(status_code=400, detail="Models not loaded")

    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Track processing step for UI updates
        update_counter = [0]  # Use list for mutable closure

        # Define incremental update callback
        async def on_incremental_update(result, img):
            """Send incremental update via WebSocket."""
            annotated = annotate_image(img, result)
            img_base64 = encode_image_base64(annotated)

            # Determine current step based on update count
            update_counter[0] += 1
            if update_counter[0] == 1:
                step = "Global scan"
            else:
                tile_num = update_counter[0] - 1
                step = f"Tile {tile_num}/9"

            await broadcast_message({
                "type": "incremental_update",
                "frame_id": file.filename,
                "step": step,
                "image": f"data:image/jpeg;base64,{img_base64}",
                "results": {
                    "anomalies": [a.to_dict() for a in result.all_anomalies],
                    "vlm_judged_count": len(result.vlm_judged_anomalies),
                    "vlm_pending": result.vlm_pending,
                    "timing_ms": result.total_time_ms,
                    "sam_candidates": result.sam_candidate_count,
                }
            })

        # Process with incremental updates
        result = await state.processor.process_single_image(
            image,
            frame_id=file.filename,
            on_update=on_incremental_update
        )

        # Draw final annotations
        annotated = annotate_image(image, result)
        img_base64 = encode_image_base64(annotated)

        response_data = {
            "status": "ok",
            "image": f"data:image/jpeg;base64,{img_base64}",
            "results": {
                "anomalies": [a.to_dict() for a in result.all_anomalies],
                "structures": [
                    {"class_name": s.class_name, "confidence": s.confidence, "bbox": s.bbox.to_xyxy()}
                    for s in result.structures
                ],
                "vlm_judged_count": len(result.vlm_judged_anomalies),
                "vlm_pending": result.vlm_pending,
                "timing_ms": result.total_time_ms,
            }
        }

        # If VLM is still pending, schedule background task
        if result.vlm_pending:
            asyncio.create_task(
                send_vlm_update_when_ready(state.processor, result, image, file.filename)
            )

        return response_data

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


async def send_vlm_update_when_ready(processor, result, image: np.ndarray, filename: str):
    """Background task: await VLM completion and send WebSocket update."""
    try:
        updated_result = await processor.await_vlm_update(result, image)

        if updated_result and len(updated_result.all_anomalies) > len(result.all_anomalies):
            # VLM added new detections - send update via WebSocket
            annotated = annotate_image(image, updated_result)
            img_base64 = encode_image_base64(annotated)

            await broadcast_message({
                "type": "vlm_update",
                "frame_id": filename,
                "image": f"data:image/jpeg;base64,{img_base64}",
                "results": {
                    "anomalies": [a.to_dict() for a in updated_result.all_anomalies],
                    "vlm_judged_count": len(updated_result.vlm_judged_anomalies),
                    "vlm_time_ms": updated_result.vlm_time_ms,
                },
                "message": f"VLM added {len(updated_result.all_anomalies) - len(result.all_anomalies)} new detections"
            })
            logger.info(f"Sent VLM update for {filename}: {len(updated_result.all_anomalies)} total anomalies")
        else:
            # VLM finished but no new detections
            await broadcast_message({
                "type": "vlm_complete",
                "frame_id": filename,
                "vlm_time_ms": updated_result.vlm_time_ms if updated_result else 0,
                "message": "VLM analysis complete, no additional detections"
            })

    except Exception as e:
        logger.error(f"VLM update task failed: {e}")
        await broadcast_message({
            "type": "vlm_error",
            "frame_id": filename,
            "error": str(e)
        })


@app.post("/api/process/video")
async def process_video(file: UploadFile = File(...)):
    """Upload and start processing a video file."""
    global state

    if state.processor is None or not state.processor.is_loaded():
        raise HTTPException(status_code=400, detail="Models not loaded")

    if state.is_processing:
        raise HTTPException(status_code=400, detail="Already processing")

    try:
        # Save uploaded video
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)

        video_path = upload_dir / f"{uuid.uuid4()}_{file.filename}"

        with open(video_path, "wb") as f:
            contents = await file.read()
            f.write(contents)

        state.current_video_path = str(video_path)

        # Start processing in background
        asyncio.create_task(process_video_task(str(video_path)))

        return {"status": "ok", "message": "Video processing started", "video_id": video_path.stem}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def process_video_task(video_path: str):
    """Background task for video processing using SAM3 Video Tracker."""
    global state

    state.is_processing = True
    state.frames_processed = 0
    state.total_detections = 0
    state.results_buffer = []

    try:
        # Get video metadata
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        await broadcast_message({
            "type": "video_start",
            "fps": fps,
            "total_frames": total_frames,
        })

        # Process video using processor.process_video() with video tracker
        # FrameSource already imported at top of file

        # Open video capture for reading frames to annotate
        cap = cv2.VideoCapture(video_path)

        # Process video with video tracker - this yields FrameResult for each frame
        async for result in state.processor.process_video(source_path=video_path, source_type=FrameSource.FILE):
            if not state.is_processing:
                break

            state.frames_processed = result.frame_index + 1
            state.total_detections += len(result.all_anomalies)

            # Read corresponding frame for annotation
            cap.set(cv2.CAP_PROP_POS_FRAMES, result.frame_index)
            ret, frame = cap.read()

            if not ret:
                logger.warning(f"Failed to read frame {result.frame_index} for annotation")
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Draw annotations
            annotated = frame_rgb.copy()

            for anomaly in result.all_anomalies:
                if anomaly.mask is not None:
                    annotated = draw_mask_overlay(annotated, anomaly.mask.data, color=(255, 0, 0), alpha=0.4)

                cv2.rectangle(
                    annotated,
                    (anomaly.bbox.x_min, anomaly.bbox.y_min),
                    (anomaly.bbox.x_max, anomaly.bbox.y_max),
                    (255, 0, 0),
                    2,
                )

            for struct in result.structures:
                cv2.rectangle(
                    annotated,
                    (struct.bbox.x_min, struct.bbox.y_min),
                    (struct.bbox.x_max, struct.bbox.y_max),
                    (0, 255, 0),
                    2,
                )

            # Encode frame
            annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
            _, buffer = cv2.imencode('.jpg', annotated_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
            img_base64 = base64.b64encode(buffer).decode('utf-8')

            # Broadcast to clients IMMEDIATELY
            await broadcast_message({
                "type": "frame",
                "frame_index": result.frame_index,
                "image": f"data:image/jpeg;base64,{img_base64}",
                "anomaly_count": len(result.all_anomalies),
                "vlm_judged_count": len(result.vlm_judged_anomalies),
                "timing_ms": result.total_time_ms,
                "progress": (result.frame_index + 1) / total_frames if total_frames > 0 else 0,
            })

            logger.info(f"Sent frame {result.frame_index}/{total_frames} with {len(result.all_anomalies)} anomalies")

            # Small delay to prevent overwhelming clients
            await asyncio.sleep(0.001)

        cap.release()

        # Get final stats
        stats = state.processor.get_stats()

        await broadcast_message({
            "type": "video_complete",
            "frames_processed": state.frames_processed,
            "total_detections": state.total_detections,
            "avg_fps": stats.avg_fps,
        })

    except Exception as e:
        await broadcast_message({
            "type": "error",
            "message": str(e),
        })

    finally:
        state.is_processing = False


@app.post("/api/stop")
async def stop_processing():
    """Stop current video processing."""
    global state
    state.is_processing = False
    return {"status": "ok", "message": "Processing stopped"}


class StreamRequest(BaseModel):
    source: str  # "webcam:0" or "rtsp://..."


@app.post("/api/stream/start")
async def start_stream(request: StreamRequest):
    """Start processing from webcam or RTSP stream."""
    global state

    if state.processor is None or not state.processor.is_loaded():
        raise HTTPException(status_code=400, detail="Models not loaded")

    if state.is_processing:
        raise HTTPException(status_code=400, detail="Already processing")

    source = request.source

    # Parse source
    if source.startswith("webcam:"):
        webcam_id = int(source.split(":")[1])
        asyncio.create_task(process_stream_task(webcam_id, is_webcam=True))
    elif source.startswith("rtsp://"):
        asyncio.create_task(process_stream_task(source, is_webcam=False))
    else:
        raise HTTPException(status_code=400, detail="Invalid source. Use 'webcam:0' or 'rtsp://...'")

    return {"status": "ok", "message": f"Stream started: {source}"}


async def process_stream_task(source, is_webcam: bool = False):
    """Background task for stream processing."""
    global state

    state.is_processing = True
    state.frames_processed = 0
    state.total_detections = 0

    try:
        # Open stream
        if is_webcam:
            cap = cv2.VideoCapture(source)
        else:
            cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            await broadcast_message({
                "type": "error",
                "message": f"Failed to open stream: {source}",
            })
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        await broadcast_message({
            "type": "stream_start",
            "fps": fps,
            "source": str(source),
        })

        frame_idx = 0
        last_time = time.time()

        while cap.isOpened() and state.is_processing:
            ret, frame = cap.read()
            if not ret:
                await asyncio.sleep(0.01)
                continue

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process frame
            result = await state.processor.process_single_image(
                frame_rgb,
                frame_id=f"stream_{frame_idx:06d}"
            )

            state.frames_processed = frame_idx + 1
            state.total_detections += len(result.all_anomalies)

            # Draw annotations
            annotated = frame_rgb.copy()

            for anomaly in result.all_anomalies:
                if anomaly.mask is not None:
                    annotated = draw_mask_overlay(annotated, anomaly.mask.data, color=(255, 0, 0), alpha=0.4)

                cv2.rectangle(
                    annotated,
                    (anomaly.bbox.x_min, anomaly.bbox.y_min),
                    (anomaly.bbox.x_max, anomaly.bbox.y_max),
                    (255, 0, 0),
                    2,
                )
                label = f"{anomaly.defect_type}: {anomaly.detection_confidence:.0%}"
                cv2.putText(annotated, label, (anomaly.bbox.x_min, anomaly.bbox.y_min - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            for struct in result.structures:
                cv2.rectangle(
                    annotated,
                    (struct.bbox.x_min, struct.bbox.y_min),
                    (struct.bbox.x_max, struct.bbox.y_max),
                    (0, 255, 0),
                    2,
                )

            # Calculate FPS
            current_time = time.time()
            actual_fps = 1.0 / (current_time - last_time) if (current_time - last_time) > 0 else 0
            last_time = current_time

            # Encode frame
            annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
            _, buffer = cv2.imencode('.jpg', annotated_bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])
            img_base64 = base64.b64encode(buffer).decode('utf-8')

            # Broadcast to clients
            await broadcast_message({
                "type": "frame",
                "frame_index": frame_idx,
                "image": f"data:image/jpeg;base64,{img_base64}",
                "anomaly_count": len(result.all_anomalies),
                "vlm_judged_count": len(result.vlm_judged_anomalies),
                "timing_ms": result.total_time_ms,
                "fps": round(actual_fps, 1),
            })

            frame_idx += 1

            # Small delay to prevent overwhelming
            await asyncio.sleep(0.001)

        cap.release()

        await broadcast_message({
            "type": "stream_stop",
            "frames_processed": state.frames_processed,
            "total_detections": state.total_detections,
        })

    except Exception as e:
        await broadcast_message({
            "type": "error",
            "message": str(e),
        })

    finally:
        state.is_processing = False


@app.post("/api/video/track/add_point")
async def add_point_to_track(request: dict):
    """
    Add a point prompt to create a new track in video processing.

    Request body:
    {
        "frame": <base64 image>,
        "frame_idx": <int>,
        "point_x": <int>,
        "point_y": <int>,
        "defect_type": <str>,
        "is_positive": <bool> (optional, default True)
    }
    """
    global state

    if state.processor is None or not state.processor.is_loaded():
        raise HTTPException(status_code=400, detail="Models not loaded")

    try:
        # Decode frame from base64
        import base64
        frame_data = base64.b64decode(request["frame"].split(",")[1] if "," in request["frame"] else request["frame"])
        frame_array = np.frombuffer(frame_data, dtype=np.uint8)
        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Add point prompt to video tracker
        track_id = state.processor.add_point_to_video_track(
            frame=frame_rgb,
            frame_idx=request["frame_idx"],
            point_xy=(request["point_x"], request["point_y"]),
            defect_type=request["defect_type"],
            is_positive=request.get("is_positive", True),
        )

        if track_id is not None:
            return {
                "status": "ok",
                "track_id": track_id,
                "message": f"Track {track_id} created successfully"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to create track")

    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing required field: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/video/tracks")
async def get_video_tracks():
    """Get all active video tracks."""
    global state

    if state.processor is None or not state.processor.is_loaded():
        raise HTTPException(status_code=400, detail="Models not loaded")

    try:
        tracks = state.processor.get_video_tracks()

        # Convert TrackingState objects to dicts
        tracks_data = []
        for track in tracks:
            tracks_data.append({
                "track_id": track.track_id,
                "defect_type": track.defect_type,
                "is_active": track.is_active,
                "frame_count": len(track.frame_indices),
                "first_frame": track.frame_indices[0] if track.frame_indices else None,
                "last_frame": track.frame_indices[-1] if track.frame_indices else None,
                "avg_confidence": sum(track.confidences) / len(track.confidences) if track.confidences else 0.0,
            })

        return {
            "status": "ok",
            "tracks": tracks_data,
            "total_tracks": len(tracks_data)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/video/track/{track_id}")
async def deactivate_track(track_id: int):
    """Deactivate a specific video track."""
    global state

    if state.processor is None or not state.processor.is_loaded():
        raise HTTPException(status_code=400, detail="Models not loaded")

    try:
        state.processor.deactivate_video_track(track_id)

        return {
            "status": "ok",
            "message": f"Track {track_id} deactivated"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await websocket.accept()
    active_connections.append(websocket)

    try:
        while True:
            # Keep connection alive, handle any incoming messages
            data = await websocket.receive_text()

            # Handle ping/pong
            if data == "ping":
                await websocket.send_text("pong")

    except WebSocketDisconnect:
        if websocket in active_connections:
            active_connections.remove(websocket)


# Mount static files
static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
