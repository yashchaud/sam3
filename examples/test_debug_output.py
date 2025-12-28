"""
Test script to verify debug output system.

This creates a simple test with a few frames to verify the debug output
system is working correctly without processing a full video.
"""

import asyncio
import numpy as np
from pathlib import Path

from anomaly_detection.realtime import RealtimeVideoProcessor, RealtimeConfig
from anomaly_detection.vlm import VLMConfig, VLMProvider


async def main():
    """Test debug output with a simple configuration."""

    # Create a simple test image (640x480 RGB)
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Configure with minimal settings
    config = RealtimeConfig(
        segmenter_model_path=Path("sam3"),  # Will load from HuggingFace
        enable_vlm_judge=False,  # Disable VLM for quick test
        vlm_config=VLMConfig(provider=VLMProvider.LOCAL_QWEN),
        device="cpu",  # Use CPU for compatibility
    )

    print("=" * 80)
    print("DEBUG OUTPUT SYSTEM TEST")
    print("=" * 80)
    print()
    print("This test will:")
    print("  1. Process a test image")
    print("  2. Save debug outputs to 'output/test_debug'")
    print("  3. Verify all output directories are created")
    print()
    print("Expected output directories:")
    print("  - 00_summary/         (Session summary)")
    print("  - 01_frames/          (Original frame)")
    print("  - 02_sam3_candidates/ (SAM3 masks)")
    print("  - 03_vlm_grids/       (VLM grid - skipped, VLM disabled)")
    print("  - 04_vlm_responses/   (VLM responses - skipped, VLM disabled)")
    print("  - 05_approved_detections/ (Final detections)")
    print("  - 06_timing/          (Timing stats)")
    print()
    print("-" * 80)

    # Process with debug output enabled
    debug_dir = Path("output/test_debug")

    async with RealtimeVideoProcessor(config, debug_output_dir=debug_dir) as processor:
        print(f"\nProcessing test image...")
        result = await processor.process_single_image(test_image, frame_id="test_frame_000001")

        print(f"\nResults:")
        print(f"  Frame ID: {result.frame_id}")
        print(f"  SAM3 candidates: {result.sam_candidate_count}")
        print(f"  Segmentation time: {result.segmentation_time_ms:.1f}ms")
        print(f"  Total time: {result.total_time_ms:.1f}ms")

    # Verify output directories
    print("\n" + "-" * 80)
    print("Verifying output directories...")

    expected_dirs = [
        "00_summary",
        "01_frames",
        "02_sam3_candidates",
        "03_vlm_grids",
        "04_vlm_responses",
        "05_approved_detections",
        "06_timing",
    ]

    all_exist = True
    for dir_name in expected_dirs:
        dir_path = debug_dir / dir_name
        exists = dir_path.exists()
        status = "✓" if exists else "✗"
        print(f"  {status} {dir_name}/")
        if not exists:
            all_exist = False

    print()
    if all_exist:
        print("SUCCESS: All debug output directories created!")
    else:
        print("WARNING: Some directories missing (may be expected if stages were skipped)")

    # Check for summary file
    summary_file = debug_dir / "00_summary" / "session_summary.json"
    if summary_file.exists():
        print(f"\n✓ Session summary created: {summary_file}")
        print(f"  File size: {summary_file.stat().st_size} bytes")
    else:
        print(f"\n✗ Session summary not found: {summary_file}")

    print("\n" + "=" * 80)
    print(f"Debug output saved to: {debug_dir.absolute()}")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
