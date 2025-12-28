"""
Example: Process a single image with full pipeline.

Demonstrates:
1. Loading and processing a single image
2. RF-DETR detection
3. SAM3 segmentation
4. Optional VLM guidance
5. Result visualization
"""

import asyncio
import argparse
from pathlib import Path
import json

from anomaly_detection.realtime import RealtimeVideoProcessor, RealtimeConfig, FrameSource
from anomaly_detection.vlm import VLMConfig, VLMProvider
from anomaly_detection.utils import load_image, save_image, draw_detections, draw_mask_overlay


def parse_args():
    parser = argparse.ArgumentParser(description="Process single image")

    parser.add_argument("image", type=str, help="Input image path")
    parser.add_argument("--sam-model", type=str, required=True, help="SAM3 model path")
    parser.add_argument("--detector-weights", type=str, default=None)
    parser.add_argument("--enable-vlm", action="store_true")
    parser.add_argument("--openrouter-key", type=str, default=None)
    parser.add_argument("--output", type=str, default="output.jpg")
    parser.add_argument("--json-output", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--debug-output", type=str, default=None, help="Enable debug output (saves all pipeline stages)")

    return parser.parse_args()


async def main():
    args = parse_args()

    # Load image
    print(f"Loading image: {args.image}")
    image = load_image(args.image)
    print(f"Image size: {image.shape[1]}x{image.shape[0]}")

    # Configure
    vlm_config = VLMConfig(
        provider=VLMProvider.OPENROUTER if args.openrouter_key else VLMProvider.LOCAL_QWEN,
        openrouter_api_key=args.openrouter_key,
    )

    config = RealtimeConfig(
        segmenter_model_path=Path(args.sam_model),
        detector_weights=Path(args.detector_weights) if args.detector_weights else None,
        enable_vlm_judge=args.enable_vlm,
        vlm_config=vlm_config,
        device=args.device,
    )

    # Process
    print("Processing...")
    if args.debug_output:
        print(f"Debug output: {args.debug_output}")

    async with RealtimeVideoProcessor(config, debug_output_dir=args.debug_output) as processor:
        result = await processor.process_single_image(image, frame_id="single_image")

    # Report results
    print(f"\nResults:")
    print(f"  Structures found: {len(result.structures)}")
    print(f"  Anomalies found: {len(result.anomalies)}")
    print(f"  VLM-judged anomalies: {len(result.vlm_judged_anomalies)}")
    print(f"  Processing time: {result.total_time_ms:.1f}ms")

    for anomaly in result.all_anomalies:
        print(f"\n  {anomaly.defect_type}:")
        print(f"    Confidence: {anomaly.combined_confidence:.1%}")
        print(f"    Location: {anomaly.bbox.to_xyxy()}")
        if anomaly.geometry:
            print(f"    Area: {anomaly.geometry.area_pixels} pixels")

    # Visualize
    output = image.copy()

    # Draw masks
    for anomaly in result.all_anomalies:
        if anomaly.mask:
            output = draw_mask_overlay(output, anomaly.mask.data, color=(255, 0, 0), alpha=0.3)

    # Draw boxes
    all_dets = [
        type('Detection', (), {
            'class_name': a.defect_type,
            'confidence': a.detection_confidence,
            'bbox': a.bbox,
        })()
        for a in result.all_anomalies
    ]
    output = draw_detections(output, all_dets, color=(255, 0, 0))

    # Draw structures
    struct_dets = [
        type('Detection', (), {
            'class_name': s.class_name,
            'confidence': s.confidence,
            'bbox': s.bbox,
        })()
        for s in result.structures
    ]
    output = draw_detections(output, struct_dets, color=(0, 255, 0))

    # Save
    save_image(output, args.output)
    print(f"\nOutput saved to: {args.output}")

    # JSON output
    if args.json_output:
        output_data = {
            "image": args.image,
            "anomalies": [a.to_dict() for a in result.all_anomalies],
            "structures": [
                {
                    "class_name": s.class_name,
                    "confidence": s.confidence,
                    "bbox": s.bbox.to_xyxy(),
                }
                for s in result.structures
            ],
            "timing_ms": result.total_time_ms,
        }
        with open(args.json_output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"JSON saved to: {args.json_output}")


if __name__ == "__main__":
    asyncio.run(main())
