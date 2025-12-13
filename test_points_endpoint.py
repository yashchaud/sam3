"""
Test the REST API points endpoint to verify set_geometric_prompt works
"""
import requests
import base64
import json

API_URL = "http://localhost:8000"

# Create a small 10x10 red test image
from PIL import Image
import io

img = Image.new('RGB', (10, 10), color='red')
img_byte_arr = io.BytesIO()
img.save(img_byte_arr, format='PNG')
img_byte_arr = img_byte_arr.getvalue()

# Test with base64 endpoint
image_base64 = base64.b64encode(img_byte_arr).decode()

data = {
    "image_base64": image_base64,
    "points_x": [5],
    "points_y": [5],
    "labels": [1],
    "return_visualization": False
}

print("Testing REST API /segment/points/base64 endpoint...")
print(f"Sending request to {API_URL}/segment/points/base64")

try:
    response = requests.post(f"{API_URL}/segment/points/base64", json=data, timeout=30)
    print(f"Status code: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print("✓ SUCCESS!")
        print(f"Response: {json.dumps(result, indent=2)}")
    else:
        print(f"✗ FAILED")
        print(f"Response: {response.text}")

except Exception as e:
    print(f"✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
