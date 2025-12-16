# SAM3 WebSocket Server Setup

Quick setup guide for the SAM3 real-time segmentation WebSocket server.

## Prerequisites

1. **Hugging Face Account**: Sign up at https://huggingface.co
2. **SAM3 Access**: Request access at https://huggingface.co/facebook/sam3
3. **Get Token**: Create a token at https://huggingface.co/settings/tokens

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Hugging Face Token

**Option A: Environment Variable (Recommended)**
```bash
export HF_TOKEN="your_token_here"
```

**Option B: Using .env file**
```bash
cp .env.example .env
# Edit .env and add your token
```

**Option C: Login via CLI**
```bash
huggingface-cli login
```

### 3. Install SAM3

```bash
git clone https://github.com/facebookresearch/sam3.git sam3_repo
cd sam3_repo && pip install -e . && cd ..
```

### 4. Start Server

```bash
python app.py
```

Server will start at: `http://localhost:8000`

## Usage

### Web Interface

Open: `http://localhost:8000/app`

### WebSocket Endpoint

Connect to: `ws://localhost:8000/ws/realtime`

**Protocol:**

1. Send init with image:
```json
{"type": "init", "image": "base64_encoded_image"}
```

2. Add points:
```json
{"type": "click", "x": 100, "y": 200, "label": 1}
```

3. Server responds with mask:
```json
{"type": "segmentation_result", "mask": "base64_png", "num_objects": 1}
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `HF_TOKEN` | Hugging Face token | Required |
| `HUGGING_FACE_HUB_TOKEN` | Alternative token variable | Optional |

## Troubleshooting

### "No Hugging Face token found"
Set the `HF_TOKEN` environment variable before starting the server.

### "Tracker model loading failed"
Make sure you have:
1. Requested access to facebook/sam3 on Hugging Face
2. Set valid HF_TOKEN
3. Installed transformers==5.0.0rc1

### "Native model loading failed"
This is expected and safe to ignore. The native model is optional and not needed for WebSocket functionality.

## Quick Test

```bash
# Set token
export HF_TOKEN="your_token_here"

# Start server
python app.py

# Open web interface
xdg-open http://localhost:8000/app  # Linux
open http://localhost:8000/app      # macOS
start http://localhost:8000/app     # Windows
```
