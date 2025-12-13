"""
Test script for real-time WebSocket segmentation
"""
import asyncio
import websockets
import json
import base64
import sys

async def test_websocket():
    """Test the WebSocket real-time segmentation endpoint"""
    uri = "ws://localhost:8000/ws/realtime"

    print(f"Connecting to {uri}...")

    try:
        async with websockets.connect(uri) as websocket:
            print("✓ Connected successfully")

            # Wait for session creation
            response = await websocket.recv()
            data = json.loads(response)

            if data.get("type") == "session_created":
                print(f"✓ Session created: {data['session_id']}")
            else:
                print(f"✗ Unexpected response: {data}")
                return

            # Test with a small test image (1x1 pixel red image)
            test_image_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="

            print("\nSending test image...")
            await websocket.send(json.dumps({
                "type": "init",
                "image": f"data:image/png;base64,{test_image_base64}"
            }))

            response = await websocket.recv()
            data = json.loads(response)

            if data.get("type") == "image_loaded":
                print(f"✓ Image loaded: {data['width']}x{data['height']}")
            elif data.get("type") == "error":
                print(f"✗ Error loading image: {data['message']}")
                return
            else:
                print(f"? Unexpected response: {data}")

            print("\nSending click point...")
            await websocket.send(json.dumps({
                "type": "click",
                "x": 0,
                "y": 0,
                "label": 1
            }))

            # Receive point_added confirmation
            response = await websocket.recv()
            data = json.loads(response)

            if data.get("type") == "point_added":
                print(f"✓ Point added: ({data['x']}, {data['y']}) - Total: {data['total_points']}")
            else:
                print(f"? Response: {data}")

            # May receive segmentation result
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                data = json.loads(response)

                if data.get("type") == "segmentation_result":
                    print(f"✓ Segmentation result received")
                    print(f"  - Confidence: {data.get('score', 0)*100:.1f}%")
                    print(f"  - Mask size: {len(data.get('mask', ''))} bytes (base64)")
                elif data.get("type") == "error":
                    print(f"✗ Segmentation error: {data['message']}")
                else:
                    print(f"? Response: {data}")
            except asyncio.TimeoutError:
                print("⚠ No segmentation result received (might be expected if model not loaded)")

            print("\nTesting reset...")
            await websocket.send(json.dumps({"type": "reset"}))

            response = await websocket.recv()
            data = json.loads(response)

            if data.get("type") == "points_cleared":
                print("✓ Points cleared")
            else:
                print(f"? Response: {data}")

            print("\nClosing connection...")
            await websocket.send(json.dumps({"type": "close"}))

            print("✓ All tests passed!")

    except websockets.exceptions.WebSocketException as e:
        print(f"✗ WebSocket error: {e}")
        print("\nMake sure the server is running:")
        print("  python app.py")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    print("SAM3 Real-time WebSocket Test")
    print("=" * 50)
    asyncio.run(test_websocket())
