# SAM3 Deployment Options

Complete guide for deploying SAM3 segmentation in different scenarios.

## 1. WebSocket Server (Real-time Interactive)

**Best for:** Web applications, interactive segmentation, multi-user

**Features:**
- Real-time segmentation as you click
- Multi-object support
- Session management
- WebSocket bi-directional communication

**Deploy:**
```bash
python app.py
```

**Endpoint:** `ws://localhost:8000/ws/realtime`

**Docs:** [FRONTEND_GUIDE.md](FRONTEND_GUIDE.md)

---

## 2. RunPod Serverless (Cost-effective, Auto-scaling)

**Best for:** API integrations, variable workloads, pay-per-use

**Features:**
- Auto-scaling (0 to N workers)
- Pay only for execution time
- No server management
- Global edge deployment

**Deploy:**
1. Push to GitHub (auto-builds image)
2. Deploy to RunPod with image URL
3. Done!

**Endpoint:** `https://api.runpod.ai/v2/YOUR_ID/run`

**Docs:** [SERVERLESS_QUICKSTART.md](SERVERLESS_QUICKSTART.md)

**Cost:** ~$0.0001 per request (RTX 4090)

---

## 3. Docker Container (Self-hosted)

**Best for:** On-premise, private cloud, consistent workload

**Features:**
- Full control
- No external dependencies
- GPU acceleration
- Custom networking

**Deploy:**
```bash
docker build -t sam3-server .
docker run --gpus all -p 8000:8000 -e HF_TOKEN=your_token sam3-server
```

**Endpoint:** `http://localhost:8000`

---

## 4. Kubernetes (Enterprise, High-availability)

**Best for:** Production deployments, high availability, complex infrastructure

**Features:**
- Auto-healing
- Load balancing
- Rolling updates
- Resource management

**Deploy:** (Coming soon)

---

## Comparison

| Feature | WebSocket Server | RunPod Serverless | Docker | Kubernetes |
|---------|-----------------|-------------------|--------|------------|
| **Setup Time** | 5 min | 10 min | 5 min | 2 hours |
| **Cost** | VM/GPU cost | Pay-per-use | VM/GPU cost | Complex |
| **Scaling** | Manual | Auto | Manual | Auto |
| **Management** | Low | None | Medium | High |
| **Use Case** | Interactive | API/Batch | Self-hosted | Enterprise |
| **Cold Start** | None | 5-10s | None | Depends |

---

## Quick Decision Guide

**Choose WebSocket Server if:**
- Building interactive web app
- Need real-time feedback
- Users click to segment
- Single deployment region

**Choose RunPod Serverless if:**
- Building API service
- Variable/unpredictable load
- Want pay-per-use pricing
- Need global deployment

**Choose Docker if:**
- Have existing infrastructure
- Need on-premise deployment
- Consistent high workload
- Full control required

**Choose Kubernetes if:**
- Enterprise production
- Need high availability
- Complex microservices
- Multiple regions

---

## Cost Examples (Monthly)

### Scenario 1: Web App (1000 users/day)
- **WebSocket**: 1x RTX 4090 24/7 = ~$290/month
- **Serverless**: 1000 req/day × 30 = ~$3/month ✅ Best

### Scenario 2: API Service (Variable load)
- **WebSocket**: 1x RTX 4090 24/7 = ~$290/month
- **Serverless**: Pay only when used = $5-50/month ✅ Best

### Scenario 3: High Volume (100K requests/day)
- **WebSocket**: 3x RTX 4090 24/7 = ~$870/month ✅ Best
- **Serverless**: 100K × 30 = ~$300/month

### Scenario 4: On-premise Required
- **Docker**: Your GPU cost ✅ Only option

---

## Getting Started

1. **For quick testing:** Run WebSocket server locally
2. **For production API:** Deploy to RunPod Serverless
3. **For enterprise:** Plan Kubernetes deployment

---

## Files Overview

```
sam3/
├── app.py                          # WebSocket server
├── serverless_handler.py           # RunPod handler
├── Dockerfile.serverless           # Serverless container
├── FRONTEND_GUIDE.md              # WebSocket client guide
├── SERVERLESS_QUICKSTART.md       # RunPod deploy guide
└── SERVERLESS_DEPLOYMENT.md       # Full serverless docs
```

---

## Next Steps

1. **Local Testing**: Start with `python app.py`
2. **Production**: Deploy serverless to RunPod
3. **Enterprise**: Contact for Kubernetes setup

---

## Support

- WebSocket issues: Check [FRONTEND_GUIDE.md](FRONTEND_GUIDE.md)
- Serverless issues: Check [SERVERLESS_DEPLOYMENT.md](SERVERLESS_DEPLOYMENT.md)
- General: Open GitHub issue
