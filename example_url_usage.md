# SAM3 API - Image URL Examples

Complete examples for using the SAM3 API with image URLs.

## API Request and Response Examples

### 1. Text Segmentation with Image URL

**Request:**
```http
POST /segment/text/url HTTP/1.1
Host: localhost:8000
Content-Type: application/json

{
  "image_url": "https://images.unsplash.com/photo-1552053831-71594a27632d?w=800",
  "text": "person",
  "return_visualization": false,
  "mask_threshold": 0.5
}
```

**Python Example:**
```python
import requests

url = "http://localhost:8000/segment/text/url"

payload = {
    "image_url": "https://images.unsplash.com/photo-1552053831-71594a27632d?w=800",
    "text": "person",
    "return_visualization": False,
    "mask_threshold": 0.5
}

response = requests.post(url, json=payload)
result = response.json()

print(f"Success: {result['success']}")
print(f"Found {result['num_masks']} objects")
print(f"Scores: {result['scores']}")

# Save masks
import base64
from PIL import Image
import io

for i, mask_b64 in enumerate(result['masks']):
    mask_data = base64.b64decode(mask_b64)
    mask_image = Image.open(io.BytesIO(mask_data))
    mask_image.save(f"mask_{i}.png")
```

**Response:**
```json
{
  "success": true,
  "num_masks": 2,
  "masks": [
    "iVBORw0KGgoAAAANSUhEUgAA...(base64 PNG data)...==",
    "iVBORw0KGgoAAAANSUhEUgAA...(base64 PNG data)...=="
  ],
  "boxes": [
    [120, 180, 450, 680],
    [500, 200, 750, 650]
  ],
  "scores": [0.96, 0.89],
  "prompt": "person"
}
```

### 2. Box Segmentation with Image URL

**Request:**
```http
POST /segment/box/url HTTP/1.1
Host: localhost:8000
Content-Type: application/json

{
  "image_url": "https://example.com/photo.jpg",
  "x1": 100,
  "y1": 150,
  "x2": 500,
  "y2": 450,
  "return_visualization": false
}
```

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/segment/box/url" \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://images.unsplash.com/photo-1518791841217-8f162f1e1131?w=800",
    "x1": 100,
    "y1": 150,
    "x2": 500,
    "y2": 450,
    "return_visualization": false
  }'
```

**Response:**
```json
{
  "success": true,
  "num_masks": 1,
  "masks": [
    "iVBORw0KGgoAAAANSUhEUgAA...(base64 PNG data)...=="
  ],
  "boxes": [[100, 150, 500, 450]],
  "scores": [0.94],
  "prompt": null
}
```

### 3. Point Segmentation with Image URL

**Request:**
```http
POST /segment/points/url HTTP/1.1
Host: localhost:8000
Content-Type: application/json

{
  "image_url": "https://example.com/image.png",
  "points_x": [300, 400],
  "points_y": [200, 250],
  "labels": [1, 1],
  "return_visualization": false
}
```

**JavaScript Example:**
```javascript
async function segmentWithPoints() {
    const response = await fetch('http://localhost:8000/segment/points/url', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            image_url: 'https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=800',
            points_x: [300, 400],
            points_y: [200, 250],
            labels: [1, 1],  // Both positive points
            return_visualization: false
        })
    });

    const result = await response.json();
    console.log(`Found ${result.num_masks} masks`);
    console.log('Scores:', result.scores);

    return result;
}

segmentWithPoints();
```

**Response:**
```json
{
  "success": true,
  "num_masks": 1,
  "masks": [
    "iVBORw0KGgoAAAANSUhEUgAA...(base64 PNG data)...=="
  ],
  "boxes": null,
  "scores": [0.91],
  "prompt": null
}
```

### 4. Automatic Segmentation with Image URL

**Request:**
```http
POST /segment/auto/url HTTP/1.1
Host: localhost:8000
Content-Type: application/json

{
  "image_url": "https://example.com/scene.jpg",
  "return_visualization": false,
  "points_per_side": 32,
  "pred_iou_thresh": 0.88,
  "stability_score_thresh": 0.95
}
```

**Python Example:**
```python
import requests

url = "http://localhost:8000/segment/auto/url"

payload = {
    "image_url": "https://picsum.photos/800/600",
    "return_visualization": False,
    "points_per_side": 32,
    "pred_iou_thresh": 0.88,
    "stability_score_thresh": 0.95
}

response = requests.post(url, json=payload)
result = response.json()

print(f"Found {result['num_masks']} objects automatically")
if result['scores']:
    avg_score = sum(result['scores']) / len(result['scores'])
    print(f"Average confidence: {avg_score:.2f}")
```

**Response:**
```json
{
  "success": true,
  "num_masks": 15,
  "masks": [
    "iVBORw0KGgoAAAANSUhEUgAA...(base64)...==",
    "iVBORw0KGgoAAAANSUhEUgAA...(base64)...==",
    "(... 13 more masks ...)"
  ],
  "boxes": null,
  "scores": [0.96, 0.94, 0.92, 0.91, 0.88, 0.87, 0.85, 0.82, 0.81, 0.79, 0.78, 0.76, 0.74, 0.72, 0.71],
  "prompt": null
}
```

### 5. Getting Visualization Instead of Masks

**Request:**
```http
POST /segment/text/url HTTP/1.1
Host: localhost:8000
Content-Type: application/json

{
  "image_url": "https://example.com/dog.jpg",
  "text": "dog",
  "return_visualization": true
}
```

**Python Example:**
```python
import requests
import base64
from PIL import Image
import io

url = "http://localhost:8000/segment/text/url"

payload = {
    "image_url": "https://images.unsplash.com/photo-1543466835-00a7907e9de1?w=800",
    "text": "dog",
    "return_visualization": True
}

response = requests.post(url, json=payload)
result = response.json()

# When return_visualization=true, the visualization is in masks[0]
viz_data = base64.b64decode(result['masks'][0])
viz_image = Image.open(io.BytesIO(viz_data))
viz_image.save("visualization.png")

print("Visualization saved to visualization.png")
```

**Response:**
```json
{
  "success": true,
  "num_masks": 1,
  "masks": [
    "iVBORw0KGgoAAAANSUhEUgAA...(base64 PNG visualization image)...=="
  ],
  "boxes": null,
  "scores": null,
  "prompt": null,
  "message": "Visualization returned in masks[0]"
}
```

## Complete Endpoint List

All endpoints accept JSON with `image_url` field:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/segment/text/url` | POST | Segment with text prompt |
| `/segment/box/url` | POST | Segment within bounding box |
| `/segment/points/url` | POST | Segment using point prompts |
| `/segment/auto/url` | POST | Automatic segmentation |

## Error Responses

### Invalid URL
```json
{
  "detail": "Failed to download image from URL: HTTPStatusError"
}
```

### Timeout
```json
{
  "detail": "Timeout while downloading image from URL"
}
```

### Image Too Large
```json
{
  "detail": "Image too large (75.3MB). Maximum size is 50MB."
}
```

### Invalid Image Format
```json
{
  "detail": "Invalid image file: cannot identify image file"
}
```

## Notes

- Maximum image size: 50MB
- Download timeout: 30 seconds
- Images >4096x4096 are automatically resized
- Supports all common image formats (JPEG, PNG, WebP, etc.)
- Works with any publicly accessible image URL
- HTTPS and HTTP URLs supported

## Real-World Examples

### Unsplash Images
```python
# Segment people in street photography
requests.post("http://localhost:8000/segment/text/url", json={
    "image_url": "https://images.unsplash.com/photo-1552053831-71594a27632d?w=800",
    "text": "person"
})

# Segment cats
requests.post("http://localhost:8000/segment/text/url", json={
    "image_url": "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=800",
    "text": "cat"
})
```

### Lorem Picsum
```python
# Random image auto-segmentation
requests.post("http://localhost:8000/segment/auto/url", json={
    "image_url": "https://picsum.photos/800/600",
    "points_per_side": 16
})
```

### Your Own Images
```python
# Use your own hosted images
requests.post("http://localhost:8000/segment/text/url", json={
    "image_url": "https://mywebsite.com/images/photo.jpg",
    "text": "car"
})
```

## Integration Examples

### Web Application
```javascript
// React/Vue/Angular frontend
async function analyzeImage(imageUrl) {
    const response = await fetch('http://localhost:8000/segment/text/url', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            image_url: imageUrl,
            text: document.getElementById('prompt').value,
            return_visualization: true
        })
    });

    const data = await response.json();

    // Display visualization
    document.getElementById('result').src =
        'data:image/png;base64,' + data.masks[0];
}
```

### Webhook/Automation
```python
# Process images from webhook
def webhook_handler(request):
    image_url = request.json['image_url']

    result = requests.post("http://localhost:8000/segment/text/url", json={
        "image_url": image_url,
        "text": "product",
        "return_visualization": False
    }).json()

    # Process masks
    for i, mask in enumerate(result['masks']):
        save_to_database(image_url, i, mask, result['scores'][i])
```
