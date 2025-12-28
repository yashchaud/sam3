#!/usr/bin/env python
"""
Launch the Anomaly Detection web interface.

Usage:
    python run_web.py [--host HOST] [--port PORT]

Example:
    python run_web.py --port 8080
"""

import argparse
import webbrowser
import threading
import time


def main():
    parser = argparse.ArgumentParser(description="Anomaly Detection Web Interface")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser")
    args = parser.parse_args()

    # Open browser after short delay
    if not args.no_browser:
        def open_browser():
            time.sleep(1.5)
            webbrowser.open(f"http://{args.host}:{args.port}")

        threading.Thread(target=open_browser, daemon=True).start()

    print(f"""
╔═══════════════════════════════════════════════════════════╗
║          🔍 Anomaly Detection Web Interface               ║
╠═══════════════════════════════════════════════════════════╣
║                                                           ║
║  Server running at: http://{args.host}:{args.port:<5}                  ║
║                                                           ║
║  Instructions:                                            ║
║  1. Enter your SAM model path in the config               ║
║  2. Click "Load Models" to initialize                     ║
║  3. Upload an image or video to process                   ║
║                                                           ║
║  Press Ctrl+C to stop the server                          ║
╚═══════════════════════════════════════════════════════════╝
    """)

    # Run server
    import uvicorn
    from web.server import app

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
