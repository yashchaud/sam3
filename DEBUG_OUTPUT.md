# Debug Output System

The debug output system saves intermediate results from each processing stage to help analyze and optimize pipeline performance.

## Overview

When enabled, the system creates a structured output directory with all pipeline stages:

```
debug_output/
├── 00_summary/           # Session summary with bottleneck analysis
├── 01_frames/            # Original input frames
├── 02_sam3_candidates/   # SAM3 segmentation masks (per defect type)
├── 03_vlm_grids/         # VLM grid overlays
├── 04_vlm_responses/     # VLM judgment JSON responses
├── 05_approved_detections/ # Final approved detections with visualization
└── 06_timing/            # Per-frame timing statistics
```

## Usage

### Command Line Examples

**Process a video with debug output:**
```bash
python examples/run_realtime_video.py \
  --source video.mp4 \
  --sam-model sam3 \
  --enable-vlm \
  --openrouter-key sk-or-v1-xxx \
  --debug-output output/debug_video
```

**Process a single image with debug output:**
```bash
python examples/run_single_image.py \
  image.jpg \
  --sam-model sam3 \
  --enable-vlm \
  --openrouter-key sk-or-v1-xxx \
  --debug-output output/debug_image
```

**Process from webcam with debug output:**
```bash
python examples/run_realtime_video.py \
  --webcam 0 \
  --sam-model sam3 \
  --enable-vlm \
  --openrouter-key sk-or-v1-xxx \
  --debug-output output/debug_webcam
```

### Programmatic Usage

```python
from anomaly_detection.realtime import RealtimeVideoProcessor, RealtimeConfig
from pathlib import Path

config = RealtimeConfig(
    segmenter_model_path=Path("sam3"),
    enable_vlm_judge=True,
    # ... other config
)

# Enable debug output by passing debug_output_dir
async with RealtimeVideoProcessor(config, debug_output_dir="output/debug") as processor:
    async for result in processor.process_video():
        # Process results
        pass

# Debug summary is automatically saved on exit
```

## Output Structure

### 1. Original Frames (`01_frames/`)
- **Files**: `frame_XXXXXX_<frame_id>.jpg`
- **Content**: Original input frames as JPG images
- **Purpose**: Reference for comparing with processed results

### 2. SAM3 Candidates (`02_sam3_candidates/`)
- **Structure**: `frame_XXXXXX/<defect_type>_mask.png` and `<defect_type>_overlay.jpg`
- **Content**:
  - Binary masks for each defect type detected by SAM3
  - Overlay visualizations with colored masks
- **Purpose**: Analyze SAM3 segmentation quality per defect class

**Summary JSON** (`frame_XXXXXX_summary.json`):
```json
{
  "frame_id": "frame_000001",
  "frame_index": 1,
  "timestamp": 1234567890.123,
  "sam3_candidates_count": 5,
  "defect_types_found": ["Crack", "Corrosion", "Spalling"],
  "processing_time_ms": 123.45
}
```

### 3. VLM Grid Overlays (`03_vlm_grids/`)
- **Files**: `frame_XXXXXX_<frame_id>_grid.jpg`
- **Content**: Image with 3x3 grid overlay (used by VLM for spatial reference)
- **Purpose**: Verify VLM's spatial understanding

### 4. VLM Responses (`04_vlm_responses/`)
- **Files**: `response_frame_XXXXXX_<frame_id>.json`
- **Content**: VLM judgments with latency and frame delay information

**Example JSON**:
```json
{
  "frame_id": "frame_000001",
  "response_frame_index": 15,
  "request_frame_index": 10,
  "frame_delay": 5,
  "timestamp": 1234567890.123,
  "latency_ms": 234.56,
  "predictions_count": 3,
  "predictions": [
    {
      "defect_type": "Crack",
      "confidence": 0.85,
      "prediction_type": "POINT",
      "point": [320, 240],
      "grid_cell": [1, 1]
    }
  ]
}
```

### 5. Approved Detections (`05_approved_detections/`)
- **Files**: `frame_XXXXXX_<frame_id>_approved.jpg` and `frame_XXXXXX_<frame_id>_metadata.json`
- **Content**:
  - Visualization of final approved detections
  - JSON metadata with bounding boxes and confidences
- **Purpose**: Final results after VLM filtering

**Metadata JSON**:
```json
{
  "frame_id": "frame_000001",
  "frame_index": 1,
  "sam_candidates_count": 5,
  "approved_count": 2,
  "approval_rate": 0.4,
  "detections": [
    {
      "defect_type": "Crack",
      "vlm_confidence": 0.85,
      "sam3_confidence": 0.75,
      "bbox": [100, 150, 200, 250]
    }
  ]
}
```

### 6. Timing Statistics (`06_timing/`)
- **Files**: `frame_XXXXXX_timing.json`
- **Content**: Per-frame timing breakdown

**Example JSON**:
```json
{
  "frame_id": "frame_000001",
  "frame_index": 1,
  "timestamp": 1234567890.123,
  "timing": {
    "sam3_ms": 123.45,
    "vlm_ms": 234.56,
    "total_ms": 358.01
  },
  "counts": {
    "sam3_candidates": 5,
    "vlm_approved": 2
  },
  "efficiency": {
    "sam3_ms_per_candidate": 24.69,
    "approval_rate": 0.4
  }
}
```

### 7. Session Summary (`00_summary/session_summary.json`)
- **File**: `session_summary.json`
- **Content**: Overall statistics and bottleneck analysis
- **Generated**: Automatically on processor exit

**Example Summary**:
```json
{
  "session_info": {
    "total_frames": 100,
    "output_directory": "output/debug",
    "timestamp": 1234567890.123
  },
  "timing_totals": {
    "sam3_total_ms": 12345.67,
    "vlm_total_ms": 23456.78,
    "total_processing_ms": 35802.45
  },
  "timing_averages": {
    "sam3_avg_ms": 123.46,
    "vlm_avg_ms": 234.57,
    "total_avg_ms": 358.02,
    "fps": 2.79
  },
  "detection_stats": {
    "total_sam3_candidates": 500,
    "total_vlm_approved": 200,
    "overall_approval_rate": 0.4,
    "avg_candidates_per_frame": 5.0,
    "avg_approved_per_frame": 2.0
  },
  "bottleneck_analysis": {
    "sam3_percentage": 34.5,
    "vlm_percentage": 65.5,
    "recommendation": "VLM is the bottleneck (65.5%). Consider: increasing VLM_EVERY_N_FRAMES, using faster model, or batching."
  }
}
```

## Analyzing Performance

### Bottleneck Identification

The summary automatically identifies performance bottlenecks:

- **SAM3 bottleneck (>70% of time)**:
  - Reduce number of defect classes
  - Use smaller SAM3 model variant
  - Enable GPU optimization

- **VLM bottleneck (>70% of time)**:
  - Increase `VLM_EVERY_N_FRAMES` (process less frequently)
  - Use faster VLM model (e.g., smaller seed model)
  - Implement batching for multiple frames

- **Balanced (<70% each)**:
  - Pipeline is well-optimized
  - Focus on overall throughput improvements

### Approval Rate Analysis

Low approval rates (<30%) may indicate:
- SAM3 generating too many false positives
- VLM confidence threshold too high
- Defect classes not well-aligned

High approval rates (>90%) may indicate:
- Could reduce VLM frequency (already high quality)
- SAM3 is doing most of the work effectively

### Latency Analysis

Check `04_vlm_responses/` for frame delays:
- **frame_delay**: Difference between request and response frame indices
- High delays (>60 frames) indicate VLM is too slow for real-time processing

## Console Output

When debug output is enabled, you'll see:

```
[Debug Output] Enabled - Output directory: output/debug
[Debug Output] Subdirectories created:
  - frames: 01_frames
  - sam3_candidates: 02_sam3_candidates
  - vlm_grids: 03_vlm_grids
  - vlm_responses: 04_vlm_responses
  - approved_detections: 05_approved_detections
  - timing: 06_timing
  - summary: 00_summary
```

On exit, a summary is printed:

```
================================================================================
DEBUG OUTPUT SUMMARY
================================================================================
Total frames processed: 100
Output directory: output/debug

TIMING ANALYSIS:
  SAM3:  12345.7ms total, 123.5ms avg (34.5%)
  VLM:   23456.8ms total, 234.6ms avg (65.5%)
  Total: 35802.5ms total, 358.0ms avg
  FPS:   2.79

DETECTION STATS:
  SAM3 candidates:  500 total, 5.0 avg/frame
  VLM approved:     200 total, 2.0 avg/frame
  Approval rate:    40.0%

BOTTLENECK:
  VLM is the bottleneck (65.5%). Consider: increasing VLM_EVERY_N_FRAMES, using faster model, or batching.
================================================================================
```

## Tips

1. **Start with a few frames**: Test debug output on 10-20 frames first to verify it's working
2. **Use for optimization**: Compare timing before/after changes
3. **Check approval rates**: Helps tune confidence thresholds
4. **Visualize results**: Open overlay images to verify detection quality
5. **Analyze JSON programmatically**: Use Python scripts to analyze timing trends

## Disabling Debug Output

Simply omit the `--debug-output` argument or don't pass `debug_output_dir` to the processor.
