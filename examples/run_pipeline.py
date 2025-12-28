"""
Example usage of the Phase-1 anomaly detection pipeline.

This script demonstrates how to:
    1. Configure the pipeline with RF-DETR and SAM3
    2. Process a single image
    3. Access and use the results

Models:
    - RF-DETR Medium: 54.7 mAP, 4.52ms latency, Apache 2.0 license
    - SAM3: 848M params, SAM license (commercial permitted)

Source:
    - RF-DETR: https://github.com/roboflow/rf-detr
    - SAM3: https://github.com/facebookresearch/sam3
"""

from pathlib import Path
import json

from anomaly_detection.pipeline import Phase1Pipeline, PipelineConfig
from anomaly_detection.detector import RFDETRVariant
from anomaly_detection.utils import load_image


def main():
    # ─────────────────────────────────────────────────────────────────
    # Configuration
    # ─────────────────────────────────────────────────────────────────
    #
    # RF-DETR uses COCO pretrained weights by default (no download needed).
    # For custom fine-tuned models, specify detector_weights and
    # detector_num_classes.

    config = PipelineConfig(
        # SAM3 model path (required)
        # Request access: https://huggingface.co/facebook/sam3
        segmenter_model_path=Path("weights/sam3.pt"),

        # RF-DETR variant: NANO, SMALL, MEDIUM (default), or LARGE
        detector_variant=RFDETRVariant.MEDIUM,

        # Optional: Custom fine-tuned detector weights
        # detector_weights=Path("weights/rfdetr_structural.pth"),
        # detector_num_classes=22,

        # Pipeline settings
        device="auto",  # Uses CUDA if available
        save_masks=True,
        mask_output_dir=Path("output/masks"),
    )

    # ─────────────────────────────────────────────────────────────────
    # Process using context manager (recommended)
    # ─────────────────────────────────────────────────────────────────
    print("Loading models...")
    print(f"  Detector: RF-DETR {config.detector_variant.value}")
    print(f"  Segmenter: SAM3")

    with Phase1Pipeline(config) as pipeline:
        print("Models loaded.\n")

        # Load and process image
        image_path = Path("test_images/bridge_inspection.jpg")
        print(f"Processing: {image_path}")

        image = load_image(image_path)
        result = pipeline.process(
            image,
            frame_id="bridge_001",
            source_path=image_path,
        )

        # ─────────────────────────────────────────────────────────────
        # Print summary
        # ─────────────────────────────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"RESULTS: {result.frame_id}")
        print(f"{'='*60}")
        print(f"Image size: {result.image_width} x {result.image_height}")
        print(f"Processing time: {result.processing_time_ms:.1f}ms")
        print(f"  - Detection:    {result.detector_time_ms:.1f}ms")
        print(f"  - Segmentation: {result.segmenter_time_ms:.1f}ms")
        print(f"\nStructures detected: {len(result.structures)}")
        print(f"Anomalies detected:  {result.anomaly_count}")

        # ─────────────────────────────────────────────────────────────
        # Print detected structures
        # ─────────────────────────────────────────────────────────────
        if result.structures:
            print(f"\n{'─'*60}")
            print("STRUCTURES:")
            print(f"{'─'*60}")

            for i, struct in enumerate(result.structures, 1):
                print(f"  [{i}] {struct.class_name} "
                      f"(confidence: {struct.confidence:.1%})")

        # ─────────────────────────────────────────────────────────────
        # Print detected anomalies
        # ─────────────────────────────────────────────────────────────
        if result.anomalies:
            print(f"\n{'─'*60}")
            print("ANOMALIES:")
            print(f"{'─'*60}")

            for i, anomaly in enumerate(result.anomalies, 1):
                print(f"\n[{i}] {anomaly.anomaly_id}")
                print(f"    Type:        {anomaly.defect_type}")
                print(f"    Structure:   {anomaly.structure_type or 'Unassociated'}")
                print(f"    Bbox:        ({anomaly.bbox.x_min:.0f}, {anomaly.bbox.y_min:.0f}) -> "
                      f"({anomaly.bbox.x_max:.0f}, {anomaly.bbox.y_max:.0f})")
                print(f"    Area:        {anomaly.geometry.area_pixels:,} pixels")
                print(f"    Dimensions:  {anomaly.geometry.length_pixels:.1f} x "
                      f"{anomaly.geometry.width_pixels:.1f} px")
                print(f"    Orientation: {anomaly.geometry.orientation_degrees:.1f} deg")
                print(f"    Confidence:  {anomaly.combined_confidence:.1%}")
                if anomaly.mask.mask_path:
                    print(f"    Mask saved:  {anomaly.mask.mask_path}")

        # ─────────────────────────────────────────────────────────────
        # Export to JSON
        # ─────────────────────────────────────────────────────────────
        output_path = Path("output/results.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        print(f"\n{'─'*60}")
        print(f"Results exported to: {output_path}")

    # Pipeline automatically unloads when exiting context manager
    print("\nPipeline unloaded.")


if __name__ == "__main__":
    main()
