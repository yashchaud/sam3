# Recent Changes - Debug Output System

## Summary

Implemented a comprehensive debug output system to help analyze and optimize pipeline performance. The system saves intermediate results from each processing stage to organized folders with automatic bottleneck analysis.

## Changes Made

### 1. New File: `anomaly_detection/debug_output.py`
- Created `DebugOutputManager` class
- Saves 7 categories of debug information:
  1. Original frames
  2. SAM3 candidate masks (individual + overlays)
  3. VLM grid overlays
  4. VLM responses (JSON)
  5. Approved detections (visualization + metadata)
  6. Per-frame timing statistics
  7. Session summary with bottleneck analysis

**Key Features**:
- Automatic bottleneck identification (SAM3 vs VLM)
- Approval rate tracking
- Latency statistics
- Optimization recommendations
- Context manager support (auto-saves summary on exit)

### 2. Modified: `anomaly_detection/realtime/realtime_processor.py`
- Added `debug_output_dir` parameter to `__init__()`
- Integrated debug calls throughout `_process_frame()`:
  - `save_original_frame()` - Input frames
  - `save_sam3_candidate()` - Individual SAM3 masks
  - `save_sam3_summary()` - SAM3 processing summary
  - `save_vlm_grid()` - VLM grid overlay (NEW)
  - `save_vlm_response()` - VLM judgments
  - `save_approved_detections()` - Final results
  - `save_frame_timing()` - Timing breakdown

**VLM Grid Integration** (Lines 312-317):
```python
# Generate grid overlay for debug output
self._vlm_judge.grid.compute_grid(frame.image.shape[1], frame.image.shape[0])
grid_image = self._vlm_judge.grid.draw_grid(frame.image)

# Debug: Save VLM grid overlay
self._debug.save_vlm_grid(frame.frame_id, frame.frame_index, grid_image)
```

### 3. Modified: `examples/run_realtime_video.py`
- Added `--debug-output` argument
- Pass `debug_output_dir` to processor
- Print debug output location when enabled

**Usage**:
```bash
python examples/run_realtime_video.py \
  --source video.mp4 \
  --sam-model sam3 \
  --enable-vlm \
  --debug-output output/debug
```

### 4. Modified: `examples/run_single_image.py`
- Added `--debug-output` argument
- Pass `debug_output_dir` to processor
- Print debug output location when enabled

**Usage**:
```bash
python examples/run_single_image.py \
  image.jpg \
  --sam-model sam3 \
  --enable-vlm \
  --debug-output output/debug
```

### 5. New File: `examples/test_debug_output.py`
- Test script to verify debug output system
- Creates simple test image
- Verifies all output directories are created
- Checks for summary file

**Usage**:
```bash
python examples/test_debug_output.py
```

### 6. New File: `DEBUG_OUTPUT.md`
- Comprehensive documentation for debug output system
- Output structure explanation
- Usage examples (CLI and programmatic)
- Performance analysis guide
- Bottleneck identification tips
- JSON format documentation

## Debug Output Structure

```
debug_output/
├── 00_summary/           # Session summary with bottleneck analysis
│   └── session_summary.json
├── 01_frames/            # Original input frames
│   └── frame_XXXXXX_<frame_id>.jpg
├── 02_sam3_candidates/   # SAM3 segmentation masks
│   ├── frame_XXXXXX/
│   │   ├── <defect_type>_mask.png
│   │   └── <defect_type>_overlay.jpg
│   └── frame_XXXXXX_summary.json
├── 03_vlm_grids/         # VLM grid overlays
│   └── frame_XXXXXX_<frame_id>_grid.jpg
├── 04_vlm_responses/     # VLM judgment JSON responses
│   └── response_frame_XXXXXX_<frame_id>.json
├── 05_approved_detections/ # Final approved detections
│   ├── frame_XXXXXX_<frame_id>_approved.jpg
│   └── frame_XXXXXX_<frame_id>_metadata.json
└── 06_timing/            # Per-frame timing statistics
    └── frame_XXXXXX_timing.json
```

## Bottleneck Analysis

The system automatically analyzes performance and provides recommendations:

### SAM3 Bottleneck (>70% of time)
**Recommendations**:
- Reduce number of defect classes
- Use smaller SAM3 model variant
- Enable GPU optimization

### VLM Bottleneck (>70% of time)
**Recommendations**:
- Increase `VLM_EVERY_N_FRAMES` (process less frequently)
- Use faster VLM model (e.g., smaller seed model)
- Implement batching for multiple frames

### Balanced Pipeline (<70% each)
**Status**: Well-optimized, focus on overall throughput

## Console Output Example

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

... (processing frames) ...

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

## Testing

To verify the debug output system works:

```bash
# Quick test with simple configuration
python examples/test_debug_output.py

# Real-world test with image
python examples/run_single_image.py \
  test_image.jpg \
  --sam-model sam3 \
  --debug-output output/test

# Check output directory
ls -la output/test/
```

## Integration Notes

- Debug output is **optional** - system works normally without it
- Minimal performance impact when disabled (`enabled=False`)
- Automatically generates summary on exit (context manager)
- All timestamps and frame indices are preserved for analysis
- JSON files are human-readable and programmatically analyzable

## Next Steps

Recommended workflow:
1. Run with debug output on a few frames to verify pipeline
2. Analyze bottlenecks in session summary
3. Adjust configuration based on recommendations
4. Re-run to verify improvements
5. Disable debug output for production use

## Files Changed

**New Files**:
- `anomaly_detection/debug_output.py` (414 lines)
- `DEBUG_OUTPUT.md` (documentation)
- `examples/test_debug_output.py` (test script)
- `CHANGES.md` (this file)

**Modified Files**:
- `anomaly_detection/realtime/realtime_processor.py`
  - Added debug system integration (8 debug calls)
  - Lines 21, 93, 276, 294-301, 312-317, 326-333, 359-366, 374-383
- `examples/run_realtime_video.py`
  - Added `--debug-output` argument
  - Lines 48, 100-101, 103
- `examples/run_single_image.py`
  - Added `--debug-output` argument
  - Lines 33, 62-63, 65

**Total Lines Added**: ~700+ lines (including documentation)
