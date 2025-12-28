"""
Example: Real-time video processing with VLM-guided anomaly detection.

This demonstrates the full pipeline:
1. Video input (file, webcam, or RTSP)
2. RF-DETR detection for structures and anomalies
3. SAM3 segmentation for pixel-accurate masks
4. VLM (Qwen3 VL) judge for additional anomaly guidance
5. Async processing with stale prediction discard
"""

import asyncio
import argparse
from pathlib import Path

from anomaly_detection.realtime import (
    RealtimeVideoProcessor,
    RealtimeConfig,
    FrameSource,
)
from anomaly_detection.vlm import VLMConfig, VLMProvider, GridConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time anomaly detection")

    # Input source
    parser.add_argument("--source", type=str, help="Video file path or RTSP URL")
    parser.add_argument("--webcam", type=int, default=None, help="Webcam ID")

    # Model paths
    parser.add_argument("--sam-model", type=str, required=True, help="SAM3 model path")
    parser.add_argument("--detector-weights", type=str, default=None, help="RF-DETR weights")

    # VLM config
    parser.add_argument("--enable-vlm", action="store_true", help="Enable VLM judge")
    parser.add_argument("--vlm-provider", choices=["local", "openrouter"], default="local")
    parser.add_argument("--openrouter-key", type=str, default=None, help="OpenRouter API key")
    parser.add_argument("--vlm-every-n", type=int, default=10, help="Process VLM every N frames")

    # Processing config
    parser.add_argument("--fps", type=float, default=30.0, help="Target FPS")
    parser.add_argument("--confidence", type=float, default=0.3, help="Detection threshold")
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda/cpu/auto)")

    # Output
    parser.add_argument("--output-dir", type=str, default=None, help="Save annotated frames")
    parser.add_argument("--debug-output", type=str, default=None, help="Enable debug output (saves all pipeline stages)")

    return parser.parse_args()


async def main():
    args = parse_args()

    # Determine source type
    if args.webcam is not None:
        source_type = FrameSource.WEBCAM
        source_path = None
        webcam_id = args.webcam
    elif args.source:
        if args.source.startswith("rtsp://"):
            source_type = FrameSource.RTSP_STREAM
        else:
            source_type = FrameSource.VIDEO_FILE
        source_path = args.source
        webcam_id = 0
    else:
        print("Error: Must specify --source or --webcam")
        return

    # Configure VLM
    vlm_config = VLMConfig(
        provider=VLMProvider.OPENROUTER if args.vlm_provider == "openrouter" else VLMProvider.LOCAL_QWEN,
        openrouter_api_key=args.openrouter_key,
        process_every_n_frames=args.vlm_every_n,
        max_generation_frames=60,
        grid_config=GridConfig(cols=3, rows=3),
    )

    # Build config
    config = RealtimeConfig(
        source_type=source_type,
        source_path=source_path,
        webcam_id=webcam_id,
        segmenter_model_path=Path(args.sam_model),
        detector_weights=Path(args.detector_weights) if args.detector_weights else None,
        enable_vlm_judge=args.enable_vlm,
        vlm_config=vlm_config,
        target_fps=args.fps,
        confidence_threshold=args.confidence,
        device=args.device,
        save_annotated_frames=args.output_dir is not None,
        annotated_output_dir=Path(args.output_dir) if args.output_dir else None,
    )

    # Process video
    print(f"Processing: {source_path or f'webcam:{webcam_id}'}")
    print(f"VLM enabled: {args.enable_vlm}")
    if args.debug_output:
        print(f"Debug output: {args.debug_output}")

    async with RealtimeVideoProcessor(config, debug_output_dir=args.debug_output) as processor:
        frame_count = 0

        async for result in processor.process_video():
            frame_count += 1

            # Print progress
            anomaly_count = len(result.all_anomalies)
            vlm_count = len(result.vlm_judged_anomalies)

            print(
                f"\rFrame {result.frame_index}: "
                f"{anomaly_count} anomalies ({vlm_count} VLM-guided), "
                f"{result.total_time_ms:.1f}ms",
                end="",
            )

            # Show VLM predictions
            if result.vlm_response and result.vlm_response.predictions:
                print(f"\n  VLM found: {[p.defect_type for p in result.vlm_response.predictions]}")

        print(f"\n\nProcessed {frame_count} frames")

        # Show stats
        stats = processor.get_stats()
        print("\nStatistics:")
        print(f"  Average FPS: {stats.avg_fps:.1f}")
        print(f"  Detections: {stats.total_detections}")
        print(f"  VLM predictions: {stats.total_vlm_predictions}")

        if stats.vlm_stats:
            print(f"  VLM latency: {stats.vlm_stats.get('mean_latency_ms', 0):.1f}ms")
            print(f"  VLM discard rate: {stats.vlm_stats.get('discard_rate', '0%')}")


if __name__ == "__main__":
    asyncio.run(main())
