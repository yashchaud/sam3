# Frontend WebSocket Integration Guide

Complete guide for connecting your frontend to the SAM3 WebSocket server and sending/receiving segmentation data.

## Server Endpoints

- **WebSocket**: `ws://localhost:8000/ws/realtime`
- **Web Interface**: `http://localhost:8000/app`
- **Health Check**: `http://localhost:8000/health`

## WebSocket Protocol

### 1. Connect to WebSocket

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/realtime');

ws.onopen = () => {
    console.log('Connected to SAM3 server');
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    handleServerMessage(data);
};

ws.onerror = (error) => {
    console.error('WebSocket error:', error);
};

ws.onclose = () => {
    console.log('Disconnected from server');
};
```

### 2. Initialize Session with Image

**Send to server:**
```javascript
// Convert image to base64
function imageToBase64(imageFile) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = reject;
        reader.readAsDataURL(imageFile);
    });
}

// Or if you have an image element
function canvasToBase64(canvas) {
    return canvas.toDataURL('image/png');
}

// Send init message
const imageBase64 = await imageToBase64(imageFile);
ws.send(JSON.stringify({
    type: "init",
    image: imageBase64  // Can include "data:image/png;base64," prefix or just the base64 string
}));
```

**Receive from server:**
```javascript
{
    "type": "session_created",
    "session_id": "550e8400-e29b-41d4-a716-446655440000"
}

// Then:
{
    "type": "image_loaded",
    "width": 1920,
    "height": 1080
}
```

### 3. Send Click Points

**Format:**
```javascript
ws.send(JSON.stringify({
    type: "click",
    x: 500,          // X coordinate (integer, 0 to image width)
    y: 300,          // Y coordinate (integer, 0 to image height)
    label: 1         // 1 = foreground (select), 0 = background (deselect)
}));
```

**Example - Select object:**
```javascript
// Click at position (500, 300) to select
ws.send(JSON.stringify({
    type: "click",
    x: 500,
    y: 300,
    label: 1
}));
```

**Example - Deselect area:**
```javascript
// Right-click or shift-click to exclude area
ws.send(JSON.stringify({
    type: "click",
    x: 600,
    y: 400,
    label: 0
}));
```

**Receive from server:**
```javascript
// First, point confirmation:
{
    "type": "point_added",
    "x": 500,
    "y": 300,
    "label": 1
}

// Then, segmentation result:
{
    "type": "segmentation_result",
    "mask": "iVBORw0KGgoAAAANSUhEUgAA...",  // Base64 PNG image
    "num_objects": 1
}
```

### 4. Display Mask Overlay

```javascript
function displayMask(maskBase64) {
    const maskImage = new Image();
    maskImage.onload = () => {
        // Draw mask on canvas
        const canvas = document.getElementById('maskCanvas');
        const ctx = canvas.getContext('2d');

        // Clear previous mask
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Draw mask with transparency
        ctx.globalAlpha = 0.5;
        ctx.drawImage(maskImage, 0, 0, canvas.width, canvas.height);
        ctx.globalAlpha = 1.0;
    };
    maskImage.src = 'data:image/png;base64,' + maskBase64;
}

// When receiving segmentation result:
function handleServerMessage(data) {
    if (data.type === 'segmentation_result') {
        displayMask(data.mask);
        console.log(`Segmented ${data.num_objects} objects`);
    }
}
```

### 5. Reset Points

```javascript
ws.send(JSON.stringify({
    type: "reset"
}));
```

**Receive:**
```javascript
{
    "type": "reset_confirmed"
}
```

### 6. Close Session

```javascript
ws.send(JSON.stringify({
    type: "close"
}));
ws.close();
```

## Complete Example

```html
<!DOCTYPE html>
<html>
<head>
    <title>SAM3 WebSocket Example</title>
    <style>
        #container { position: relative; width: 800px; height: 600px; }
        #imageCanvas, #maskCanvas { position: absolute; top: 0; left: 0; }
        #maskCanvas { pointer-events: none; }
    </style>
</head>
<body>
    <div id="container">
        <canvas id="imageCanvas" width="800" height="600"></canvas>
        <canvas id="maskCanvas" width="800" height="600"></canvas>
    </div>
    <input type="file" id="imageInput" accept="image/*">
    <button id="resetBtn">Reset Points</button>

    <script>
        let ws = null;
        const imageCanvas = document.getElementById('imageCanvas');
        const maskCanvas = document.getElementById('maskCanvas');
        const imageCtx = imageCanvas.getContext('2d');
        const maskCtx = maskCanvas.getContext('2d');

        // Connect to WebSocket
        function connect() {
            ws = new WebSocket('ws://localhost:8000/ws/realtime');

            ws.onopen = () => {
                console.log('Connected to SAM3 server');
            };

            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                handleMessage(data);
            };

            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
            };

            ws.onclose = () => {
                console.log('Disconnected');
            };
        }

        // Handle server messages
        function handleMessage(data) {
            console.log('Received:', data.type);

            switch(data.type) {
                case 'session_created':
                    console.log('Session ID:', data.session_id);
                    break;

                case 'image_loaded':
                    console.log(`Image loaded: ${data.width}x${data.height}`);
                    break;

                case 'point_added':
                    drawPoint(data.x, data.y, data.label);
                    break;

                case 'segmentation_result':
                    displayMask(data.mask);
                    console.log(`Segmented ${data.num_objects} objects`);
                    break;

                case 'reset_confirmed':
                    maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
                    break;

                case 'error':
                    console.error('Server error:', data.message);
                    alert('Error: ' + data.message);
                    break;
            }
        }

        // Load and send image
        document.getElementById('imageInput').addEventListener('change', async (e) => {
            const file = e.target.files[0];
            if (!file) return;

            // Display image
            const img = new Image();
            img.onload = () => {
                imageCanvas.width = img.width;
                imageCanvas.height = img.height;
                maskCanvas.width = img.width;
                maskCanvas.height = img.height;
                imageCtx.drawImage(img, 0, 0);
            };
            img.src = URL.createObjectURL(file);

            // Convert to base64 and send
            const reader = new FileReader();
            reader.onload = () => {
                ws.send(JSON.stringify({
                    type: "init",
                    image: reader.result
                }));
            };
            reader.readAsDataURL(file);
        });

        // Handle clicks
        imageCanvas.addEventListener('click', (e) => {
            const rect = imageCanvas.getBoundingClientRect();
            const x = Math.round((e.clientX - rect.left) * (imageCanvas.width / rect.width));
            const y = Math.round((e.clientY - rect.top) * (imageCanvas.height / rect.height));

            ws.send(JSON.stringify({
                type: "click",
                x: x,
                y: y,
                label: e.shiftKey ? 0 : 1  // Shift+click for background
            }));
        });

        // Reset button
        document.getElementById('resetBtn').addEventListener('click', () => {
            ws.send(JSON.stringify({ type: "reset" }));
        });

        // Draw point marker
        function drawPoint(x, y, label) {
            const ctx = imageCtx;
            ctx.fillStyle = label === 1 ? 'lime' : 'red';
            ctx.beginPath();
            ctx.arc(x, y, 5, 0, 2 * Math.PI);
            ctx.fill();
            ctx.strokeStyle = 'white';
            ctx.lineWidth = 2;
            ctx.stroke();
        }

        // Display mask overlay
        function displayMask(maskBase64) {
            const maskImage = new Image();
            maskImage.onload = () => {
                maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
                maskCtx.globalAlpha = 0.6;
                maskCtx.drawImage(maskImage, 0, 0);
                maskCtx.globalAlpha = 1.0;
            };
            maskImage.src = 'data:image/png;base64,' + maskBase64;
        }

        // Connect on load
        connect();
    </script>
</body>
</html>
```

## Message Format Reference

### Client → Server

| Message Type | Fields | Example |
|-------------|--------|---------|
| `init` | `type`, `image` | `{"type": "init", "image": "data:image/png;base64,..."}` |
| `click` | `type`, `x`, `y`, `label` | `{"type": "click", "x": 100, "y": 200, "label": 1}` |
| `reset` | `type` | `{"type": "reset"}` |
| `close` | `type` | `{"type": "close"}` |

### Server → Client

| Message Type | Fields | Description |
|-------------|--------|-------------|
| `session_created` | `type`, `session_id` | Session initialized |
| `image_loaded` | `type`, `width`, `height` | Image processed successfully |
| `point_added` | `type`, `x`, `y`, `label` | Point registered |
| `segmentation_result` | `type`, `mask`, `num_objects` | Segmentation mask (base64 PNG) |
| `reset_confirmed` | `type` | Points cleared |
| `error` | `type`, `message` | Error occurred |

## Multi-Object Segmentation

The server automatically handles multiple objects:

```javascript
// Click on first object
ws.send(JSON.stringify({ type: "click", x: 100, y: 100, label: 1 }));

// Click on second object (far from first, >100px)
ws.send(JSON.stringify({ type: "click", x: 500, y: 500, label: 1 }));

// Refine first object (within 100px of existing point)
ws.send(JSON.stringify({ type: "click", x: 110, y: 110, label: 1 }));
```

**Proximity threshold**: 100 pixels
- Points within 100px → refine same object
- Points beyond 100px → create new object

## Error Handling

```javascript
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);

    if (data.type === 'error') {
        console.error('Server error:', data.message);

        // Common errors:
        // - "No image loaded" → Send init first
        // - "Point (x, y) outside image bounds" → Check coordinates
        // - "Max objects (50) reached" → Reset and start over
        // - "Max points per object (100) reached" → Start new object
    }
};
```

## Testing with curl

```bash
# Install websocat
# On macOS: brew install websocat
# On Linux: cargo install websocat

# Connect and test
websocat ws://localhost:8000/ws/realtime

# Then paste JSON messages:
{"type": "init", "image": "data:image/png;base64,iVBORw0KGgo..."}
{"type": "click", "x": 100, "y": 200, "label": 1}
{"type": "reset"}
```

## React Example

```javascript
import { useEffect, useRef, useState } from 'react';

function SAM3Client() {
    const ws = useRef(null);
    const [mask, setMask] = useState(null);
    const [connected, setConnected] = useState(false);

    useEffect(() => {
        ws.current = new WebSocket('ws://localhost:8000/ws/realtime');

        ws.current.onopen = () => setConnected(true);
        ws.current.onclose = () => setConnected(false);

        ws.current.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.type === 'segmentation_result') {
                setMask(data.mask);
            }
        };

        return () => ws.current.close();
    }, []);

    const sendClick = (x, y, label = 1) => {
        if (ws.current?.readyState === WebSocket.OPEN) {
            ws.current.send(JSON.stringify({ type: "click", x, y, label }));
        }
    };

    return (
        <div>
            <p>Status: {connected ? 'Connected' : 'Disconnected'}</p>
            {/* Your UI here */}
        </div>
    );
}
```

## Notes

- Image must be sent as base64 in `init` message
- Coordinates are in original image space (not canvas/display space)
- Mask is returned as base64 PNG (grayscale, 0=background, 255=foreground)
- All messages are JSON strings
- Server maintains session state per WebSocket connection
- Maximum 50 objects per session, 100 points per object
- Session timeout: 1 hour of inactivity
