# RunPod Serverless Quick Start

Deploy SAM3 segmentation as a serverless function in 5 minutes.

## Step 1: Push to GitHub

```bash
git add .
git commit -m "Add serverless support"
git push origin main
```

GitHub Actions will automatically build the Docker image.

## Step 2: Get Image URL

After build completes (~10 minutes), your image will be at:
```
ghcr.io/YOUR_USERNAME/YOUR_REPO/sam3-serverless:latest
```

## Step 3: Deploy to RunPod

1. Go to: https://www.runpod.io/console/serverless
2. Click "New Endpoint"
3. Fill in:
   - **Name**: `sam3-segmentation`
   - **Container Image**: Your image URL from above
   - **Container Disk**: `10 GB`
   - **GPU**: Select `RTX 4090` or `A40`
   - **Environment Variables**: Add `HF_TOKEN=your_token`
   - **Max Workers**: `3`
4. Click "Deploy"

## Step 4: Test

```bash
curl -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "image": "data:image/png;base64,iVBORw0KG...",
      "points": [[100, 200]],
      "labels": [1]
    }
  }'
```

## Input Format

```json
{
  "input": {
    "image": "base64_string_with_or_without_prefix",
    "points": [[x1, y1], [x2, y2]],
    "labels": [1, 0]
  }
}
```

- `image`: Base64 encoded image
- `points`: Array of [x, y] coordinates
- `labels`: Array of 1 (foreground) or 0 (background)

## Output Format

```json
{
  "id": "request-id",
  "status": "COMPLETED",
  "output": {
    "mask": "base64_png_string",
    "mask_shape": [height, width],
    "num_points": 2
  }
}
```

## Cost Estimate

**RTX 4090** (~$0.40/hr):
- Cold start: ~5-10 seconds
- Warm inference: ~0.5-1 second
- Cost per request: ~$0.0001-0.0002

**Example usage:**
- 1000 requests/day = ~$0.10-0.20/day
- 30,000 requests/month = ~$3-6/month

## Files Created

- `Dockerfile.serverless` - Container definition
- `serverless_handler.py` - RunPod handler
- `.github/workflows/build-serverless.yml` - Auto-build on push
- `test_serverless.py` - Local testing script

## Full Documentation

See [SERVERLESS_DEPLOYMENT.md](SERVERLESS_DEPLOYMENT.md) for complete guide.
