# Real-time SAM3 Segmentation

This guide explains how to use the real-time click/touch-based segmentation feature.

## Overview

The real-time segmentation pipeline allows you to:
- Load an image in your browser
- Click on objects to segment them instantly
- See segmentation results in real-time
- Refine segmentations with multiple clicks
- Use both foreground and background points

## Architecture

### Backend (WebSocket Server)
- **Endpoint**: `ws://localhost:8000/ws/realtime`
- **Protocol**: JSON-based bidirectional communication
- **Session Management**: Each connection gets a unique session with persistent inference state
- **Real-time Processing**: Segmentation triggered automatically on each click

### Frontend (Web Interface)
- **URL**: `http://localhost:8000/app`
- **Technology**: Pure HTML5 Canvas + JavaScript (no frameworks)
- **Features**:
  - Drag & drop or file upload for images
  - Interactive canvas with click/touch support
  - Real-time mask overlay visualization
  - Point markers showing click locations
  - Confidence score display

## Getting Started

### 1. Start the Server

Make sure SAM3 server is running:

```bash
# On Windows
start_server.bat

# On Linux/Mac
./start_server.sh

# Or with Python directly
python app.py
```

### 2. Open the Web Interface

Navigate to: **http://localhost:8000/app**

### 3. Load an Image

Click the "📁 Load Image" button and select an image from your computer.

### 4. Start Segmenting

- **Left-click** on an object to add foreground points (green markers)
- **Right-click** to add background points (red markers) for refinement
- The segmentation mask updates automatically after each click
- The confidence score shows how certain the model is about the segmentation

### 5. Refine Your Segmentation

- Add more foreground points to expand the selection
- Add background points to exclude unwanted areas
- Click "🔄 Reset Points" to clear all points and start over
- Click "🗑️ Clear Image" to load a new image

## WebSocket Protocol

### Client → Server Messages

#### Initialize Session with Image
```json
{
  "type": "init",
  "image": "data:image/png;base64,..."
}
```

#### Add Click Point
```json
{
  "type": "click",
  "x": 100,
  "y": 200,
  "label": 1  // 1 = foreground, 0 = background
}
```

#### Reset Points
```json
{
  "type": "reset"
}
```

#### Close Session
```json
{
  "type": "close"
}
```

### Server → Client Messages

#### Session Created
```json
{
  "type": "session_created",
  "session_id": "uuid-here"
}
```

#### Image Loaded
```json
{
  "type": "image_loaded",
  "width": 1920,
  "height": 1080
}
```

#### Point Added
```json
{
  "type": "point_added",
  "x": 100,
  "y": 200,
  "label": 1,
  "total_points": 3
}
```

#### Segmentation Result
```json
{
  "type": "segmentation_result",
  "mask": "base64_encoded_png",
  "score": 0.95,
  "num_points": 3
}
```

#### Points Cleared
```json
{
  "type": "points_cleared"
}
```

#### Error
```json
{
  "type": "error",
  "message": "Error description"
}
```

## Building Your Own Client

You can build custom clients using the WebSocket protocol. Here's a Python example:

```python
import websockets
import asyncio
import json
import base64

async def realtime_segmentation():
    uri = "ws://localhost:8000/ws/realtime"

    async with websockets.connect(uri) as websocket:
        # Wait for session creation
        response = await websocket.recv()
        data = json.loads(response)
        print(f"Session ID: {data['session_id']}")

        # Load image
        with open("image.jpg", "rb") as f:
            image_data = base64.b64encode(f.read()).decode()

        await websocket.send(json.dumps({
            "type": "init",
            "image": f"data:image/jpeg;base64,{image_data}"
        }))

        response = await websocket.recv()
        print(json.loads(response))

        # Add a click point
        await websocket.send(json.dumps({
            "type": "click",
            "x": 100,
            "y": 200,
            "label": 1
        }))

        # Receive results
        response = await websocket.recv()
        result = json.loads(response)

        if result["type"] == "segmentation_result":
            # Save mask
            mask_data = base64.b64decode(result["mask"])
            with open("mask.png", "wb") as f:
                f.write(mask_data)

        # Close connection
        await websocket.send(json.dumps({"type": "close"}))

asyncio.run(realtime_segmentation())
```

## Features

### Session Management
- Each WebSocket connection creates a unique session
- Sessions maintain inference state for fast repeated segmentations
- Automatic cleanup when connection closes

### Performance Optimization
- Inference state is cached per session
- No need to re-encode image for each click
- Masks are efficiently encoded as PNG images
- Automatic reconnection on disconnect

### User Interface Features
- **Responsive Design**: Works on desktop and mobile
- **Touch Support**: Touch events work on mobile devices
- **Visual Feedback**: Point markers show click locations
- **Mask Overlay**: Semi-transparent mask overlay on the image
- **Toggle Mask**: Show/hide mask overlay
- **Confidence Display**: Real-time confidence scores
- **Image Info**: Display image dimensions and session ID

## Troubleshooting

### Connection Issues

**Problem**: "Disconnected from server"
- Check if the server is running
- Verify the URL is correct (ws://localhost:8000/ws/realtime)
- Check firewall settings

### Performance Issues

**Problem**: Slow segmentation
- Ensure CUDA is enabled (check server logs)
- Try smaller images (resize before upload)
- Close other applications using GPU

### Segmentation Quality

**Problem**: Poor segmentation results
- Add more foreground points on the object
- Add background points to exclude unwanted areas
- Try clicking on different parts of the object
- Reset points and start fresh

## Technical Details

### Backend Implementation
- **File**: `app.py` (lines 1457-1657)
- **WebSocket Handler**: `websocket_realtime_segmentation()`
- **Session Storage**: `realtime_sessions` dictionary
- **Auto-cleanup**: Sessions removed on disconnect

### Frontend Implementation
- **HTML**: `static/index.html`
- **JavaScript**: `static/app.js`
- **Class**: `RealtimeSAM`
- **Canvas Rendering**: Dual canvas system (image + mask overlay)

### API Endpoints
- **WebSocket**: `/ws/realtime` - Real-time segmentation
- **Web Interface**: `/app` - HTML interface
- **Static Files**: `/static/*` - CSS/JS assets

## Advanced Usage

### Custom Mask Processing

You can process masks client-side after receiving them:

```javascript
async displayMask(maskBase64) {
    const img = new Image();
    img.onload = () => {
        // Get mask as canvas
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = img.width;
        tempCanvas.height = img.height;
        const ctx = tempCanvas.getContext('2d');
        ctx.drawImage(img, 0, 0);

        // Get pixel data
        const imageData = ctx.getImageData(0, 0, img.width, img.height);

        // Process mask (e.g., change color, apply effects)
        // ... your processing code here

        // Display processed mask
        this.maskCtx.putImageData(imageData, 0, 0);
    };
    img.src = 'data:image/png;base64,' + maskBase64;
}
```

### Multi-object Segmentation

For segmenting multiple objects:
1. Segment first object
2. Save the mask
3. Click "Reset Points"
4. Segment second object
5. Combine masks programmatically

## Next Steps

- Explore the REST API for batch processing (see [README.md](README.md))
- Try text-based segmentation for natural language queries
- Implement your own client using the WebSocket protocol
- Integrate real-time segmentation into your application

## Support

For issues or questions:
- Check the main [README.md](README.md)
- Review [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- Open an issue on GitHub
