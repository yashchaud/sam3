#!/usr/bin/env python3
"""
Quick test script to check what RF-DETR detects on an image.
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image
import requests
from io import BytesIO

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from anomaly_detection.detector import RFDETRDetector, DetectorConfig, RFDETRVariant


def test_url_image(url: str, confidence: float = 0.1):
    """Test detection on image from URL."""
    print(f"Testing detection on: {url}")
    print(f"Confidence threshold: {confidence}\n")

    # Download image
    response = requests.get(url)
    image = Image.open(BytesIO(response.content)).convert("RGB")
    image_array = np.array(image)

    print(f"Image size: {image.size}")

    # Create detector
    config = DetectorConfig(
        variant=RFDETRVariant.MEDIUM,
        confidence_threshold=confidence,
        device="cuda",
    )

    detector = RFDETRDetector(config)
    print("Loading detector...")
    detector.load()
    print("Detector loaded!\n")

    # Run detection
    print("Running detection...")
    result = detector.detect(image_array)

    print(f"\nResults:")
    print(f"  Inference time: {result.inference_time_ms:.2f}ms")
    print(f"  Total detections: {len(result.anomalies) + len(result.structures)}")
    print(f"  Anomalies: {len(result.anomalies)}")
    print(f"  Structures: {len(result.structures)}")

    if result.anomalies:
        print("\nDetected Anomalies:")
        for det in result.anomalies:
            print(f"  - {det.class_name}: {det.confidence:.3f} at [{det.bbox.x_min:.0f}, {det.bbox.y_min:.0f}, {det.bbox.x_max:.0f}, {det.bbox.y_max:.0f}]")

    if result.structures:
        print("\nDetected Structures:")
        for det in result.structures:
            print(f"  - {det.class_name}: {det.confidence:.3f} at [{det.bbox.x_min:.0f}, {det.bbox.y_min:.0f}, {det.bbox.x_max:.0f}, {det.bbox.y_max:.0f}]")

    if not result.anomalies and not result.structures:
        print("\nNo detections found!")
        print("\nThis is likely because:")
        print("1. RF-DETR is using COCO pretrained weights")
        print("2. COCO doesn't have 'crack' or 'soil defect' classes")
        print("3. You need to fine-tune RF-DETR on your specific anomaly dataset")
        print("\nCOCO classes include: person, car, dog, cat, etc. - not industrial defects")


if __name__ == "__main__":
    # Test with the soil crack image
    url = "https://cdn.pixabay.com/photo/2020/06/26/08/28/soil-5342049_640.jpg"

    # Try with very low confidence to see if anything is detected
    test_url_image(url, confidence=0.05)
