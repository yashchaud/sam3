"""
Test script for serverless handler locally
"""
import base64
from serverless_handler import handler

def load_test_image():
    """Load a test image and convert to base64"""
    try:
        with open("test_image.jpg", "rb") as f:
            return base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        print("Error: test_image.jpg not found")
        print("Please add a test image named 'test_image.jpg' in the current directory")
        return None

def test_single_point():
    """Test with a single foreground point"""
    print("\n=== Test 1: Single Point ===")
    image_b64 = load_test_image()
    if not image_b64:
        return

    job = {
        "input": {
            "image": f"data:image/jpeg;base64,{image_b64}",
            "points": [[500, 300]],
            "labels": [1]
        }
    }

    result = handler(job)

    if "error" in result:
        print(f"Error: {result['error']}")
        if "traceback" in result:
            print(result["traceback"])
    else:
        print(f"Success!")
        print(f"Mask shape: {result['mask_shape']}")
        print(f"Num points: {result['num_points']}")

        mask_bytes = base64.b64decode(result['mask'])
        with open("test_mask_single.png", "wb") as f:
            f.write(mask_bytes)
        print("Saved mask to: test_mask_single.png")

def test_multiple_points():
    """Test with multiple points"""
    print("\n=== Test 2: Multiple Points ===")
    image_b64 = load_test_image()
    if not image_b64:
        return

    job = {
        "input": {
            "image": f"data:image/jpeg;base64,{image_b64}",
            "points": [[500, 300], [520, 310], [480, 290]],
            "labels": [1, 1, 1]
        }
    }

    result = handler(job)

    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Success!")
        print(f"Mask shape: {result['mask_shape']}")

        mask_bytes = base64.b64decode(result['mask'])
        with open("test_mask_multiple.png", "wb") as f:
            f.write(mask_bytes)
        print("Saved mask to: test_mask_multiple.png")

def test_with_background():
    """Test with foreground and background points"""
    print("\n=== Test 3: Foreground + Background ===")
    image_b64 = load_test_image()
    if not image_b64:
        return

    job = {
        "input": {
            "image": f"data:image/jpeg;base64,{image_b64}",
            "points": [[500, 300], [600, 400]],
            "labels": [1, 0]  # First is foreground, second is background
        }
    }

    result = handler(job)

    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Success!")
        print(f"Mask shape: {result['mask_shape']}")

        mask_bytes = base64.b64decode(result['mask'])
        with open("test_mask_background.png", "wb") as f:
            f.write(mask_bytes)
        print("Saved mask to: test_mask_background.png")

if __name__ == "__main__":
    print("SAM3 Serverless Handler Test")
    print("=" * 50)

    test_single_point()
    test_multiple_points()
    test_with_background()

    print("\n" + "=" * 50)
    print("Tests completed!")
