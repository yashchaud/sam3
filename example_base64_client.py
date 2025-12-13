#!/usr/bin/env python3
"""
Example client demonstrating base64 image input for SAM3 Segmentation API

This script shows how to use the JSON/base64 endpoints for web applications.
"""

import requests
import base64
import json
from PIL import Image
import io
from pathlib import Path

# API Configuration
API_URL = "http://localhost:8000"

def encode_image_to_base64(image_path: str) -> str:
    """Encode an image file to base64 string"""
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    return base64.b64encode(image_bytes).decode('utf-8')

def decode_base64_to_image(base64_string: str) -> Image.Image:
    """Decode base64 string to PIL Image"""
    image_bytes = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(image_bytes))

def save_base64_image(base64_string: str, output_path: str):
    """Save a base64-encoded image to file"""
    image = decode_base64_to_image(base64_string)
    image.save(output_path)
    print(f"  Saved: {output_path}")

def segment_text_base64(image_path: str, text: str, output_dir: str = "output"):
    """Segment image using text prompt with base64 input"""
    print(f"\n=== Text Segmentation (Base64) ===")
    print(f"Image: {image_path}")
    print(f"Text prompt: '{text}'")

    Path(output_dir).mkdir(exist_ok=True)

    # Encode image to base64
    image_base64 = encode_image_to_base64(image_path)

    # Prepare JSON request
    payload = {
        "image_base64": image_base64,
        "text": text,
        "return_visualization": False,
        "mask_threshold": 0.5
    }

    # Send request
    response = requests.post(
        f"{API_URL}/segment/text/base64",
        json=payload,
        headers={"Content-Type": "application/json"}
    )

    if response.status_code == 200:
        result = response.json()
        print(f"✓ Found {result['num_masks']} objects")
        if result['scores']:
            print(f"  Scores: {[f'{s:.3f}' for s in result['scores']]}")

        # Save masks
        for i, mask_b64 in enumerate(result['masks']):
            output_path = f"{output_dir}/base64_text_mask_{text.replace(' ', '_')}_{i}.png"
            save_base64_image(mask_b64, output_path)
    else:
        print(f"✗ Error: {response.status_code}")
        print(f"  {response.json()}")

def segment_text_with_visualization(image_path: str, text: str, output_dir: str = "output"):
    """Segment with text and get visualization as base64"""
    print(f"\n=== Text Segmentation with Visualization (Base64) ===")
    print(f"Image: {image_path}")
    print(f"Text prompt: '{text}'")

    Path(output_dir).mkdir(exist_ok=True)

    # Encode image to base64
    image_base64 = encode_image_to_base64(image_path)

    # Prepare JSON request
    payload = {
        "image_base64": image_base64,
        "text": text,
        "return_visualization": True
    }

    # Send request
    response = requests.post(
        f"{API_URL}/segment/text/base64",
        json=payload,
        headers={"Content-Type": "application/json"}
    )

    if response.status_code == 200:
        result = response.json()
        print(f"✓ Received visualization")

        # Save visualization (it's in masks[0])
        output_path = f"{output_dir}/base64_viz_{text.replace(' ', '_')}.png"
        save_base64_image(result['masks'][0], output_path)
    else:
        print(f"✗ Error: {response.status_code}")
        print(f"  {response.json()}")

def segment_box_base64(image_path: str, box: list, output_dir: str = "output"):
    """Segment using bounding box with base64 input"""
    print(f"\n=== Box Segmentation (Base64) ===")
    print(f"Image: {image_path}")
    print(f"Box: {box}")

    Path(output_dir).mkdir(exist_ok=True)

    # Encode image to base64
    image_base64 = encode_image_to_base64(image_path)

    x1, y1, x2, y2 = box

    # Prepare JSON request
    payload = {
        "image_base64": image_base64,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "return_visualization": False
    }

    # Send request
    response = requests.post(
        f"{API_URL}/segment/box/base64",
        json=payload,
        headers={"Content-Type": "application/json"}
    )

    if response.status_code == 200:
        result = response.json()
        print(f"✓ Found {result['num_masks']} masks")
        if result['scores']:
            print(f"  Scores: {[f'{s:.3f}' for s in result['scores']]}")

        # Save masks
        for i, mask_b64 in enumerate(result['masks']):
            output_path = f"{output_dir}/base64_box_mask_{i}.png"
            save_base64_image(mask_b64, output_path)
    else:
        print(f"✗ Error: {response.status_code}")
        print(f"  {response.json()}")

def segment_points_base64(image_path: str, points_x: list, points_y: list,
                          labels: list, output_dir: str = "output"):
    """Segment using points with base64 input"""
    print(f"\n=== Point Segmentation (Base64) ===")
    print(f"Image: {image_path}")
    print(f"Points: {list(zip(points_x, points_y))}")
    print(f"Labels: {labels} (1=positive, 0=negative)")

    Path(output_dir).mkdir(exist_ok=True)

    # Encode image to base64
    image_base64 = encode_image_to_base64(image_path)

    # Prepare JSON request
    payload = {
        "image_base64": image_base64,
        "points_x": points_x,
        "points_y": points_y,
        "labels": labels,
        "return_visualization": False
    }

    # Send request
    response = requests.post(
        f"{API_URL}/segment/points/base64",
        json=payload,
        headers={"Content-Type": "application/json"}
    )

    if response.status_code == 200:
        result = response.json()
        print(f"✓ Found {result['num_masks']} masks")
        if result['scores']:
            print(f"  Scores: {[f'{s:.3f}' for s in result['scores']]}")

        # Save masks
        for i, mask_b64 in enumerate(result['masks']):
            output_path = f"{output_dir}/base64_points_mask_{i}.png"
            save_base64_image(mask_b64, output_path)
    else:
        print(f"✗ Error: {response.status_code}")
        print(f"  {response.json()}")

def segment_auto_base64(image_path: str, output_dir: str = "output",
                        points_per_side: int = 32):
    """Automatic segmentation with base64 input"""
    print(f"\n=== Automatic Segmentation (Base64) ===")
    print(f"Image: {image_path}")
    print(f"Points per side: {points_per_side}")

    Path(output_dir).mkdir(exist_ok=True)

    # Encode image to base64
    image_base64 = encode_image_to_base64(image_path)

    # Prepare JSON request
    payload = {
        "image_base64": image_base64,
        "return_visualization": False,
        "points_per_side": points_per_side,
        "pred_iou_thresh": 0.88,
        "stability_score_thresh": 0.95
    }

    # Send request
    response = requests.post(
        f"{API_URL}/segment/auto/base64",
        json=payload,
        headers={"Content-Type": "application/json"}
    )

    if response.status_code == 200:
        result = response.json()
        print(f"✓ Found {result['num_masks']} objects")
        if result['scores']:
            avg_score = sum(result['scores']) / len(result['scores'])
            print(f"  Average score: {avg_score:.3f}")

        # Save first 10 masks
        for i, mask_b64 in enumerate(result['masks'][:10]):
            output_path = f"{output_dir}/base64_auto_mask_{i}.png"
            save_base64_image(mask_b64, output_path)

        if len(result['masks']) > 10:
            print(f"  (Saved first 10 of {len(result['masks'])} masks)")
    else:
        print(f"✗ Error: {response.status_code}")
        print(f"  {response.json()}")

def javascript_example():
    """Print JavaScript example code"""
    print("\n" + "="*70)
    print("JavaScript Example (for web applications)")
    print("="*70)

    js_code = '''
// Convert image to base64
async function imageToBase64(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result.split(',')[1]);
        reader.onerror = reject;
        reader.readAsDataURL(file);
    });
}

// Segment with text prompt
async function segmentImage(imageFile, text) {
    const imageBase64 = await imageToBase64(imageFile);

    const response = await fetch('http://localhost:8000/segment/text/base64', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            image_base64: imageBase64,
            text: text,
            return_visualization: false,
            mask_threshold: 0.5
        })
    });

    const result = await response.json();
    console.log(`Found ${result.num_masks} objects`);

    // Convert mask base64 back to image
    result.masks.forEach((maskBase64, i) => {
        const img = new Image();
        img.src = 'data:image/png;base64,' + maskBase64;
        document.body.appendChild(img);
    });
}

// Usage
const fileInput = document.getElementById('imageInput');
fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    segmentImage(file, 'person');
});
'''
    print(js_code)
    print("="*70 + "\n")

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="SAM3 API Base64 Client Examples",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--image", help="Path to image file", required=True)
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--all", action="store_true", help="Run all examples")
    parser.add_argument("--js-example", action="store_true", help="Show JavaScript example")

    args = parser.parse_args()

    if args.js_example:
        javascript_example()
        return

    if not Path(args.image).exists():
        print(f"Error: Image file not found: {args.image}")
        return

    print(f"\n{'='*70}")
    print(f"SAM3 API - Base64 Examples")
    print(f"{'='*70}")

    # Get image dimensions
    img = Image.open(args.image)
    width, height = img.size
    print(f"\nImage size: {width}x{height}")

    if args.all:
        # Example 1: Text segmentation
        segment_text_base64(args.image, "person", args.output_dir)

        # Example 2: Text with visualization
        segment_text_with_visualization(args.image, "car", args.output_dir)

        # Example 3: Box segmentation
        box = [width//4, height//4, 3*width//4, 3*height//4]
        segment_box_base64(args.image, box, args.output_dir)

        # Example 4: Point segmentation
        segment_points_base64(
            args.image,
            points_x=[width//2],
            points_y=[height//2],
            labels=[1],
            output_dir=args.output_dir
        )

        # Example 5: Automatic segmentation
        segment_auto_base64(args.image, args.output_dir, points_per_side=16)

        # Show JavaScript example
        javascript_example()
    else:
        # Quick demo
        segment_text_with_visualization(args.image, "person", args.output_dir)

    print(f"\n{'='*70}")
    print(f"Examples completed! Check the '{args.output_dir}' folder.")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
