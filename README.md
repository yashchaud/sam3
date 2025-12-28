# Real-time Anomaly Detection Pipeline

A production-ready pipeline for structural inspection using:
- **RF-DETR** for object detection (structures & anomalies)
- **SAM3** for pixel-accurate segmentation
- **Qwen3 VL 8B** as a judge model for additional guidance
- **Async processing** with intelligent stale prediction discard

## Architecture

```
Video/Image Input
       │
       ▼
┌──────────────────┐
│  Frame Buffer    │ ◄─── Async capture
└────────┬─────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌───────┐  ┌─────────────┐
│RF-DETR│  │  VLM Judge  │ ◄─── Every N frames (async)
│Detect │  │ (Qwen3 VL)  │
└───┬───┘  └──────┬──────┘
    │             │
    │    Grid overlay + structured output
    │             │
    ▼             ▼
┌─────────────────────────┐
│       SAM3 Segment      │
│  (box or point prompts) │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│    Results + Stats      │
└─────────────────────────┘
```

## Features

- **Real-time processing**: Target 30+ FPS on modern GPUs
- **VLM-guided detection**: Qwen3 VL analyzes frames for defects the detector might miss
- **Async VLM processing**: VLM runs in background, predictions merged when ready
- **Intelligent discard**: Stale predictions (>60 frames old) are automatically discarded
- **Multiple input sources**: Video files, webcam, RTSP streams, image sequences
- **Latency tracking**: Comprehensive statistics for all pipeline stages
- **OpenRouter support**: Use cloud VLM if local GPU is limited

## Installation

```bash
# Clone and install
git clone <repo-url>
cd anomalyDetection
pip install -e ".[all]"

# Install SAM3 separately (requires Git)
pip install git+https://github.com/facebookresearch/sam3.git
```

## Quick Start

### Process Single Image

```python
import asyncio
from pathlib import Path
from anomaly_detection.realtime import RealtimeVideoProcessor, RealtimeConfig
from anomaly_detection.utils import load_image

async def main():
    config = RealtimeConfig(
        segmenter_model_path=Path("weights/sam3_hiera_large.pt"),
        enable_vlm_judge=True,
    )

    async with RealtimeVideoProcessor(config) as processor:
        image = load_image("structure.jpg")
        result = await processor.process_single_image(image)

        for anomaly in result.all_anomalies:
            print(f"{anomaly.defect_type}: {anomaly.combined_confidence:.1%}")

asyncio.run(main())
```

### Process Video

```python
import asyncio
from pathlib import Path
from anomaly_detection.realtime import RealtimeVideoProcessor, RealtimeConfig, FrameSource

async def main():
    config = RealtimeConfig(
        source_type=FrameSource.VIDEO_FILE,
        source_path="inspection.mp4",
        segmenter_model_path=Path("weights/sam3_hiera_large.pt"),
        enable_vlm_judge=True,
        vlm_process_every_n_frames=10,
    )

    async with RealtimeVideoProcessor(config) as processor:
        async for result in processor.process_video():
            print(f"Frame {result.frame_index}: {len(result.all_anomalies)} anomalies")

asyncio.run(main())
```

### Using OpenRouter for VLM

```python
from anomaly_detection.vlm import VLMConfig, VLMProvider

vlm_config = VLMConfig(
    provider=VLMProvider.OPENROUTER,
    openrouter_api_key="sk-or-...",
    openrouter_model="qwen/qwen-2.5-vl-72b-instruct",
    process_every_n_frames=10,
)

config = RealtimeConfig(
    segmenter_model_path=Path("weights/sam3.pt"),
    enable_vlm_judge=True,
    vlm_config=vlm_config,
)
```

## CLI Usage

```bash
# Single image
python -m examples.run_single_image image.jpg \
    --sam-model weights/sam3.pt \
    --enable-vlm \
    --output result.jpg

# Video processing
python -m examples.run_realtime_video \
    --source video.mp4 \
    --sam-model weights/sam3.pt \
    --enable-vlm \
    --vlm-every-n 10

# Webcam
python -m examples.run_realtime_video \
    --webcam 0 \
    --sam-model weights/sam3.pt
```

## VLM Judge System

The VLM judge uses a grid overlay to communicate defect locations:

```
┌─────┬─────┬─────┐
│ A1  │ A2  │ A3  │
├─────┼─────┼─────┤
│ B1  │ B2  │ B3  │
├─────┼─────┼─────┤
│ C1  │ C2  │ C3  │
└─────┴─────┴─────┘
```

The VLM returns structured JSON:
```json
{
  "predictions": [
    {"cell": "B2", "defect_type": "crack", "confidence": 0.85}
  ]
}
```

Grid cells are converted to either:
- **Points**: Cell center for SAM point prompting
- **Boxes**: Cell bounds for SAM box prompting

## Async Processing & Discard Logic

```
Frame 0 ────► Detection ────► Segmentation ────► Output
Frame 1 ────► Detection ────► Segmentation ────► Output
...
Frame 10 ───► Detection + VLM Request (async)
             └──────────────────────────────────────────┐
Frame 11 ───► Detection ────► Segmentation ────► Output│
...                                                     │
Frame 25 ◄──── VLM Response Ready ◄─────────────────────┘
             │
             ▼
         Merge VLM predictions into frame 25 output
```

If VLM response arrives after 60 frames, it's discarded (scene has changed).

## Statistics

```python
stats = processor.get_stats()
print(stats.to_dict())
# {
#   "frames": {"processed": 1000, "dropped": 5},
#   "detections": 342,
#   "vlm": {
#     "predictions": 45,
#     "discarded": 3,
#     "mean_latency_ms": 150.5,
#     "p95_latency_ms": 280.2,
#     "discard_rate": "6.7%"
#   },
#   "performance": {
#     "avg_fps": 28.5,
#     "avg_detection_time_ms": 12.3,
#     "avg_segmentation_time_ms": 18.7
#   }
# }
```

## Supported Defect Types

**Anomalies (defects):**
- crack, corrosion, spalling, deformation, stain
- efflorescence, exposed_rebar, delamination
- scaling, popout, honeycomb, rust

**Structures:**
- beam, column, wall, slab, pipe
- foundation, joint, girder, truss, deck

## Project Structure

```
anomaly_detection/
├── __init__.py
├── config.py              # Environment configuration
├── models/
│   └── data_models.py     # Core data structures
├── detector/
│   └── rf_detr_detector.py
├── segmenter/
│   └── sam3_segmenter.py
├── vlm/
│   ├── models.py          # VLM data models
│   ├── grid_overlay.py    # Grid system
│   ├── qwen_client.py     # Local Qwen
│   ├── openrouter_client.py
│   └── vlm_judge.py       # Unified interface
├── realtime/
│   ├── config.py          # Realtime config
│   ├── frame_buffer.py    # Thread-safe buffer
│   ├── stream_handler.py  # Video input
│   └── realtime_processor.py
├── tiling/
│   ├── tiler.py          # Image tiling
│   └── coordinator.py     # Tiled detection
├── association/
│   └── structure_defect_matcher.py
├── geometry/
│   └── mask_geometry.py
└── utils/
    └── image_utils.py
```

## License

MIT License - See LICENSE file for details.

Model licenses:
- RF-DETR: Apache 2.0
- SAM3: Apache 2.0
- Qwen: Various (check model card)
