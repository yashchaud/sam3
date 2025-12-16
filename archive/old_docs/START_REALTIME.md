# Quick Start Guide - Real-time Segmentation

## 1. Start the Server

**Windows:**
```cmd
start_server.bat
```

**Linux/Mac:**
```bash
./start_server.sh
```

**Or manually:**
```bash
python app.py
```

Wait for the message:
```
INFO:__main__:SAM3 model loaded successfully!
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Static files mounted from c:\sam3\static
```

## 2. Open the Web Interface

Navigate to: **http://localhost:8000/app**

## 3. Test the Connection

You should see:
- Status: "Connected - Ready to load image"
- Green status indicator

If you see "Connecting to server..." or red status:
1. Check the server is running
2. Check the browser console (F12) for errors
3. Verify static files are served: http://localhost:8000/static/app.js

## 4. Start Segmenting

1. Click **"📁 Load Image"** and select an image
2. Wait for "Image loaded - Click to segment"
3. **Left-click** on an object (adds green foreground point)
4. Watch the segmentation appear in real-time!
5. **Right-click** to add red background points for refinement

## Troubleshooting

### "Failed to load resource: 404"
- **Issue**: Static files not being served
- **Check**: Visit http://localhost:8000/static/app.js directly
- **Fix**: Ensure `static/` directory exists with `index.html` and `app.js`

### "Disconnected from server"
- **Issue**: WebSocket connection failed
- **Check**: Look for errors in server logs
- **Fix**: Restart the server

### "Model not loaded yet"
- **Issue**: SAM3 model still loading
- **Fix**: Wait 30-60 seconds for model to load (3.45GB download on first run)

### Stuck at "Connecting to server..."
- **Issue**: WebSocket endpoint not available
- **Check browser console**: Open DevTools (F12) → Console tab
- **Look for**: WebSocket connection errors
- **Common causes**:
  - Server not running
  - Wrong URL (should be `ws://localhost:8000/ws/realtime`)
  - Firewall blocking WebSocket connections

## Testing

Run the automated test:
```bash
python test_realtime.py
```

Expected output:
```
SAM3 Real-time WebSocket Test
==================================================
Connecting to ws://localhost:8000/ws/realtime...
✓ Connected successfully
✓ Session created: 12345678-...
✓ Image loaded: 1x1
✓ Point added: (0, 0) - Total: 1
✓ Segmentation result received
  - Confidence: 95.2%
  - Mask size: 1234 bytes (base64)
✓ Points cleared
✓ All tests passed!
```

## Browser Console Debugging

Open browser DevTools (F12) and check:

**Console tab** - Look for:
```
Connected successfully
Session created: [uuid]
Image loaded
Segmentation result received
```

**Network tab** - Check:
- WebSocket connection to `ws://localhost:8000/ws/realtime` (Status: 101 Switching Protocols)
- GET requests to `/static/app.js` (Status: 200)

## Next Steps

- See [REALTIME_SEGMENTATION.md](REALTIME_SEGMENTATION.md) for complete documentation
- See [README.md](README.md) for REST API endpoints
- Build custom clients using the WebSocket protocol
