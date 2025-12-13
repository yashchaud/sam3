# SAM3 API - Project Structure

Complete overview of all files in the SAM3 Image Segmentation API project.

## Core Application Files

### `app.py`
**Main FastAPI application** - Complete REST API server with 12 endpoints
- Text, box, points, and auto segmentation methods
- Three input modes: file upload, base64 JSON, image URL
- Model loading and inference
- Comprehensive error handling
- CORS middleware
- Health check endpoints

### `requirements.txt`
**Python dependencies** for the API service
```
fastapi==0.115.5
uvicorn[standard]==0.32.1
python-multipart==0.0.20
pillow>=10.0.0
numpy>=1.24.0
opencv-python>=4.8.0
matplotlib>=3.7.0
pydantic>=2.0.0
aiofiles>=23.0.0
httpx>=0.27.0
```

## Setup Scripts

### Linux/macOS Setup

#### `setup.sh`
**Automated setup script** for Linux/macOS
- Python version checking (requires 3.12+)
- Virtual environment creation
- pip upgrade
- API dependencies installation
- CUDA detection
- PyTorch installation (CUDA or CPU)
- SAM3 installation with HuggingFace authentication
- Output directory creation

#### `start_server.sh`
**Server starter script** for Linux/macOS
- Virtual environment activation
- SAM3 installation verification
- Server startup with clear instructions

### Windows Setup

#### `setup.bat`
**Automated setup script** for Windows
- Same functionality as setup.sh
- Windows-compatible batch script
- Colored output and error handling

#### `start_server.bat`
**Server starter script** for Windows
- Same functionality as start_server.sh
- Windows-compatible batch script

## Docker Files

### `Dockerfile`
**Docker container configuration**
- Base: nvidia/cuda:12.6.0-runtime-ubuntu22.04
- Python 3.12 installation
- PyTorch 2.7.0 with CUDA 12.6
- SAM3 installation from official repository
- API dependencies
- Port 8000 exposed

### `docker-compose.yml`
**Docker Compose configuration**
- Service definition
- GPU support configuration
- Port mapping
- Volume mounts

### `.dockerignore`
**Docker build exclusions**
- Excludes venv, output, cache files
- Reduces image size

## Documentation

### `README.md`
**Main documentation** - Complete API guide
- Quick start section
- Features overview
- Three installation methods
- All 12 endpoint examples
- Python, JavaScript, and cURL examples
- Visualization mode
- Error handling
- Configuration options
- Troubleshooting

### `QUICKSTART.md`
**3-step quick start guide**
- Prerequisites checklist
- Simple clone-and-start instructions
- Quick test examples
- Troubleshooting section
- Next steps

### `INSTALL.md`
**Detailed installation guide**
- Step-by-step instructions
- Prerequisites explained
- Local and Docker installation
- Verification steps
- Performance tips
- Common issues

### `example_url_usage.md`
**Image URL endpoint documentation**
- Complete request/response examples
- All 4 methods with URL input
- Python, JavaScript, cURL examples
- Error responses
- Real-world integration examples

### `PROJECT_STRUCTURE.md`
**This file** - Complete project overview

## Example Client Scripts

### `example_client.py`
**File upload endpoint examples**
- Text segmentation
- Box segmentation
- Point segmentation
- Automatic segmentation
- Visualization mode
- Command-line interface
- Mask saving functionality

### `example_base64_client.py`
**Base64 JSON endpoint examples**
- All 4 segmentation methods
- Base64 encoding/decoding
- JavaScript example code
- Web application integration examples
- Command-line interface

## Directory Structure

```
sam3/
├── app.py                      # Main FastAPI application
├── requirements.txt            # Python dependencies
│
├── setup.sh                    # Linux/macOS setup script
├── start_server.sh             # Linux/macOS server starter
├── setup.bat                   # Windows setup script
├── start_server.bat            # Windows server starter
│
├── Dockerfile                  # Docker container config
├── docker-compose.yml          # Docker Compose config
├── .dockerignore              # Docker build exclusions
│
├── README.md                   # Main documentation
├── QUICKSTART.md              # Quick start guide
├── INSTALL.md                 # Installation guide
├── example_url_usage.md       # URL endpoint examples
├── PROJECT_STRUCTURE.md       # This file
│
├── example_client.py          # File upload examples
├── example_base64_client.py   # Base64 JSON examples
│
├── output/                     # Created by setup - mask outputs
├── venv/                       # Created by setup - Python environment
└── sam3_repo/                  # Created by setup - SAM3 source code
```

## API Endpoints Overview

### 12 Total Endpoints (4 methods × 3 input modes)

| Method | File Upload | Base64 JSON | Image URL |
|--------|------------|-------------|-----------|
| **Text** | POST /segment/text | POST /segment/text/base64 | POST /segment/text/url |
| **Box** | POST /segment/box | POST /segment/box/base64 | POST /segment/box/url |
| **Points** | POST /segment/points | POST /segment/points/base64 | POST /segment/points/url |
| **Auto** | POST /segment/auto | POST /segment/auto/base64 | POST /segment/auto/url |

### Utility Endpoints

- `GET /` - Health check
- `GET /health` - Health check (alias)

## Usage Patterns

### For Quick Testing
1. Use `setup.sh` or `setup.bat`
2. Run `start_server.sh` or `start_server.bat`
3. Visit http://localhost:8000/docs
4. Try example scripts

### For Development
1. Manual installation (Option 2 in README)
2. Use `uvicorn app:app --reload` for auto-reload
3. Modify app.py as needed

### For Production Deployment
1. Use Docker (Option 3 in README)
2. Configure environment variables
3. Set up reverse proxy (nginx/traefik)
4. Enable HTTPS

### For Web Applications
1. Use base64 endpoints (`/segment/*/base64`)
2. See `example_base64_client.py` JavaScript examples
3. Handle CORS properly

### For Automation/Pipelines
1. Use URL endpoints (`/segment/*/url`)
2. See `example_url_usage.md`
3. Batch process images from URLs

## Getting Started

**Absolute fastest way to start:**

```bash
# Clone repository
git clone <repo-url>
cd sam3

# Linux/macOS
chmod +x setup.sh start_server.sh
./setup.sh
./start_server.sh

# Windows
setup.bat
start_server.bat
```

**First test:**
```bash
curl http://localhost:8000/health
```

**First segmentation:**
```bash
curl -X POST "http://localhost:8000/segment/text" \
  -F "image=@photo.jpg" \
  -F "text=person"
```

## Support Files

All scripts include:
- ✓ Error handling
- ✓ Colored output (where supported)
- ✓ Progress indicators
- ✓ Verification steps
- ✓ Clear instructions

## Next Steps After Setup

1. Read [QUICKSTART.md](QUICKSTART.md) for basic usage
2. Explore http://localhost:8000/docs for interactive API docs
3. Run example scripts to see all features
4. Check [README.md](README.md) for integration examples
5. Review [example_url_usage.md](example_url_usage.md) for URL API usage

## File Permissions Note

**Linux/macOS users**: Make scripts executable:
```bash
chmod +x setup.sh start_server.sh
```

**Windows users**: No special permissions needed, just run `.bat` files directly.
