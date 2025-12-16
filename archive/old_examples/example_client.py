#!/usr/bin/env python3
"""
Example client for SAM3 Segmentation API

This script demonstrates all available endpoints and features.
"""

import requests
import base64
import argparse
from PIL import Image
import io
from pathlib import Path
import sys

# API Configuration
API_URL = "http://localhost:8000"

def check_health():
    """Check if the API is running and healthy"""
    try:
        response = requests.get(f"{API_URL}/")
        if response.status_code == 200:
            data = response.json()
            print(f"✓ API is running")
            print(f"  Model: {data['model']}")
            print(f"  Model loaded: {data['model_loaded']}")
            print(f"  Device: {data['device']}")
            print(f"  Version: {data['version']}")
            return True
        else:
            print(f"✗ API returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"✗ Could not connect to API at {API_URL}")
        print(f"  Make sure the server is running: python app.py")
        return False
    except Exception as e:
        print(f"✗ Error checking health: {e}")
        return False

def save_mask(mask_b64: str, output_path: str):
    """Save a base64-encoded mask to file"""
    mask_data = base64.b64decode(mask_b64)
    mask_image = Image.open(io.BytesIO(mask_data))
    mask_image.save(output_path)
    print(f"  Saved mask: {output_path}")

def segment_text(image_path: str, text: str, output_dir: str = "output", visualize: bool = False):
    """Segment image using text prompt"""
    print(f"\n=== Text Segmentation ===")
    print(f"Image: {image_path}")
    print(f"Text prompt: '{text}'")

    Path(output_dir).mkdir(exist_ok=True)

    with open(image_path, 'rb') as f:
        files = {"image": f}
        data = {
            "text": text,
            "return_visualization": str(visualize).lower(),
            "mask_threshold": 0.5
        }

        response = requests.post(f"{API_URL}/segment/text", files=files, data=data)

    if response.status_code == 200:
        if visualize:
            # Save visualization image
            output_path = f"{output_dir}/text_viz_{text.replace(' ', '_')}.png"
            with open(output_path, 'wb') as f:
                f.write(response.content)
            print(f"✓ Visualization saved: {output_path}")
        else:
            # Save individual masks
            result = response.json()
            print(f"✓ Found {result['num_masks']} objects")
            if result['scores']:
                print(f"  Scores: {[f'{s:.3f}' for s in result['scores']]}")

            for i, mask_b64 in enumerate(result['masks']):
                output_path = f"{output_dir}/text_mask_{text.replace(' ', '_')}_{i}.png"
                save_mask(mask_b64, output_path)
    else:
        print(f"✗ Error: {response.status_code}")
        print(f"  {response.json()}")

def segment_box(image_path: str, box: list, output_dir: str = "output", visualize: bool = False):
    """Segment image using bounding box"""
    print(f"\n=== Box Segmentation ===")
    print(f"Image: {image_path}")
    print(f"Box: {box}")

    Path(output_dir).mkdir(exist_ok=True)

    x1, y1, x2, y2 = box
    with open(image_path, 'rb') as f:
        files = {"image": f}
        data = {
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "return_visualization": str(visualize).lower()
        }

        response = requests.post(f"{API_URL}/segment/box", files=files, data=data)

    if response.status_code == 200:
        if visualize:
            output_path = f"{output_dir}/box_viz.png"
            with open(output_path, 'wb') as f:
                f.write(response.content)
            print(f"✓ Visualization saved: {output_path}")
        else:
            result = response.json()
            print(f"✓ Found {result['num_masks']} masks")
            if result['scores']:
                print(f"  Scores: {[f'{s:.3f}' for s in result['scores']]}")

            for i, mask_b64 in enumerate(result['masks']):
                output_path = f"{output_dir}/box_mask_{i}.png"
                save_mask(mask_b64, output_path)
    else:
        print(f"✗ Error: {response.status_code}")
        print(f"  {response.json()}")

def segment_points(image_path: str, points_x: list, points_y: list, labels: list,
                   output_dir: str = "output", visualize: bool = False):
    """Segment image using point prompts"""
    print(f"\n=== Point Segmentation ===")
    print(f"Image: {image_path}")
    print(f"Points: {list(zip(points_x, points_y))}")
    print(f"Labels: {labels} (1=positive, 0=negative)")

    Path(output_dir).mkdir(exist_ok=True)

    with open(image_path, 'rb') as f:
        files = {"image": f}
        data = {
            "points_x": ",".join(map(str, points_x)),
            "points_y": ",".join(map(str, points_y)),
            "labels": ",".join(map(str, labels)),
            "return_visualization": str(visualize).lower()
        }

        response = requests.post(f"{API_URL}/segment/points", files=files, data=data)

    if response.status_code == 200:
        if visualize:
            output_path = f"{output_dir}/points_viz.png"
            with open(output_path, 'wb') as f:
                f.write(response.content)
            print(f"✓ Visualization saved: {output_path}")
        else:
            result = response.json()
            print(f"✓ Found {result['num_masks']} masks")
            if result['scores']:
                print(f"  Scores: {[f'{s:.3f}' for s in result['scores']]}")

            for i, mask_b64 in enumerate(result['masks']):
                output_path = f"{output_dir}/points_mask_{i}.png"
                save_mask(mask_b64, output_path)
    else:
        print(f"✗ Error: {response.status_code}")
        print(f"  {response.json()}")

def segment_auto(image_path: str, output_dir: str = "output", visualize: bool = False,
                 points_per_side: int = 32):
    """Automatically segment all objects in image"""
    print(f"\n=== Automatic Segmentation ===")
    print(f"Image: {image_path}")
    print(f"Points per side: {points_per_side}")

    Path(output_dir).mkdir(exist_ok=True)

    with open(image_path, 'rb') as f:
        files = {"image": f}
        data = {
            "return_visualization": str(visualize).lower(),
            "points_per_side": points_per_side,
            "pred_iou_thresh": 0.88,
            "stability_score_thresh": 0.95
        }

        response = requests.post(f"{API_URL}/segment/auto", files=files, data=data)

    if response.status_code == 200:
        if visualize:
            output_path = f"{output_dir}/auto_viz.png"
            with open(output_path, 'wb') as f:
                f.write(response.content)
            print(f"✓ Visualization saved: {output_path}")
        else:
            result = response.json()
            print(f"✓ Found {result['num_masks']} objects automatically")
            if result['scores']:
                avg_score = sum(result['scores']) / len(result['scores'])
                print(f"  Average score: {avg_score:.3f}")

            # Save up to 10 masks to avoid too many files
            for i, mask_b64 in enumerate(result['masks'][:10]):
                output_path = f"{output_dir}/auto_mask_{i}.png"
                save_mask(mask_b64, output_path)

            if len(result['masks']) > 10:
                print(f"  (Saved first 10 of {len(result['masks'])} masks)")
    else:
        print(f"✗ Error: {response.status_code}")
        print(f"  {response.json()}")

def run_all_examples(image_path: str):
    """Run all segmentation examples"""
    print(f"\n{'='*60}")
    print(f"SAM3 API Client - All Examples")
    print(f"{'='*60}")

    # Check health
    if not check_health():
        return

    # Get image dimensions for box example
    img = Image.open(image_path)
    width, height = img.size
    print(f"\nImage size: {width}x{height}")

    # Example 1: Text segmentation
    segment_text(image_path, "person", visualize=True)

    # Example 2: Box segmentation (center quarter of image)
    box = [width//4, height//4, 3*width//4, 3*height//4]
    segment_box(image_path, box, visualize=True)

    # Example 3: Point segmentation (center point)
    segment_points(
        image_path,
        points_x=[width//2],
        points_y=[height//2],
        labels=[1],
        visualize=True
    )

    # Example 4: Automatic segmentation
    segment_auto(image_path, visualize=True, points_per_side=16)

    print(f"\n{'='*60}")
    print(f"All examples completed! Check the 'output' folder.")
    print(f"{'='*60}\n")

def main():
    parser = argparse.ArgumentParser(
        description="SAM3 API Client - Example Usage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check API health
  python example_client.py --health

  # Run all examples
  python example_client.py --image photo.jpg --all

  # Text segmentation
  python example_client.py --image photo.jpg --text "person" --visualize

  # Box segmentation
  python example_client.py --image photo.jpg --box 100 150 500 450 --visualize

  # Point segmentation
  python example_client.py --image photo.jpg --points-x 200,300 --points-y 150,250 --labels 1,1

  # Automatic segmentation
  python example_client.py --image photo.jpg --auto --visualize
        """
    )

    parser.add_argument("--url", default="http://localhost:8000", help="API URL")
    parser.add_argument("--health", action="store_true", help="Check API health")
    parser.add_argument("--image", help="Path to image file")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--visualize", action="store_true", help="Return visualization instead of masks")

    # All examples
    parser.add_argument("--all", action="store_true", help="Run all examples")

    # Text segmentation
    parser.add_argument("--text", help="Text prompt for segmentation")

    # Box segmentation
    parser.add_argument("--box", nargs=4, type=int, metavar=("X1", "Y1", "X2", "Y2"),
                       help="Bounding box coordinates")

    # Point segmentation
    parser.add_argument("--points-x", help="Comma-separated x coordinates")
    parser.add_argument("--points-y", help="Comma-separated y coordinates")
    parser.add_argument("--labels", help="Comma-separated labels (1=positive, 0=negative)")

    # Auto segmentation
    parser.add_argument("--auto", action="store_true", help="Automatic segmentation")
    parser.add_argument("--points-per-side", type=int, default=32, help="Points per side for auto mode")

    args = parser.parse_args()

    # Update API URL
    global API_URL
    API_URL = args.url

    # Health check only
    if args.health:
        check_health()
        return

    # Require image for all other operations
    if not args.image:
        if not args.health:
            parser.print_help()
            print("\nError: --image is required")
            sys.exit(1)
        return

    # Check if image exists
    if not Path(args.image).exists():
        print(f"Error: Image file not found: {args.image}")
        sys.exit(1)

    # Run all examples
    if args.all:
        run_all_examples(args.image)
        return

    # Individual operations
    if args.text:
        segment_text(args.image, args.text, args.output_dir, args.visualize)

    if args.box:
        segment_box(args.image, args.box, args.output_dir, args.visualize)

    if args.points_x and args.points_y and args.labels:
        points_x = [int(x) for x in args.points_x.split(",")]
        points_y = [int(y) for y in args.points_y.split(",")]
        labels = [int(l) for l in args.labels.split(",")]
        segment_points(args.image, points_x, points_y, labels, args.output_dir, args.visualize)

    if args.auto:
        segment_auto(args.image, args.output_dir, args.visualize, args.points_per_side)

if __name__ == "__main__":
    main()
