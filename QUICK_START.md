# Quick Start Guide

## 1. Setup (One-time)

```bash
# Install dependencies
pip install -r requirements.txt

# Set Hugging Face token
export HF_TOKEN="your_token_here"

# Or create .env file
cp .env.example .env
# Edit .env and add your token
```

## 2. Start Server

```bash
python app.py
```

Server starts at: `http://localhost:8000`

## 3. Use WebSocket Interface

### Option A: Built-in Web Interface (Easiest)

Open in browser: `http://localhost:8000/app`

1. Upload an image
2. Click on objects to segment
3. See real-time mask overlay

### Option B: Your Own Frontend

**Connect:**
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/realtime');
```

**Send image:**
```javascript
ws.send(JSON.stringify({
    type: "init",
    image: "data:image/png;base64,..."
}));
```

**Send click:**
```javascript
ws.send(JSON.stringify({
    type: "click",
    x: 100,
    y: 200,
    label: 1  // 1=foreground, 0=background
}));
```

**Receive mask:**
```javascript
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === 'segmentation_result') {
        const maskBase64 = data.mask;  // Base64 PNG image
        // Display it
    }
};
```

## Message Format

### Client → Server

**Init with image:**
```json
{"type": "init", "image": "base64_string"}
```

**Add point:**
```json
{"type": "click", "x": 100, "y": 200, "label": 1}
```

**Reset:**
```json
{"type": "reset"}
```

### Server → Client

**Session created:**
```json
{"type": "session_created", "session_id": "uuid"}
```

**Image loaded:**
```json
{"type": "image_loaded", "width": 1920, "height": 1080}
```

**Segmentation result:**
```json
{"type": "segmentation_result", "mask": "base64_png", "num_objects": 1}
```

**Error:**
```json
{"type": "error", "message": "error description"}
```

## Full Documentation

- **Frontend Integration**: See [FRONTEND_GUIDE.md](FRONTEND_GUIDE.md)
- **Setup Details**: See [SETUP.md](SETUP.md)
- **Main README**: See [README.md](README.md)

## Example Files

- **Web UI**: `static/index.html` and `static/app.js`
- **Working Example**: Already at `http://localhost:8000/app`

## Quick Test

```bash
# 1. Start server
python app.py

# 2. Open browser
xdg-open http://localhost:8000/app  # Linux
open http://localhost:8000/app      # macOS
start http://localhost:8000/app     # Windows

# 3. Upload image and click to segment!
```
