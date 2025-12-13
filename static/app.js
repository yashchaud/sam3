class RealtimeSAM {
    constructor() {
        this.ws = null;
        this.sessionId = null;
        this.imageLoaded = false;
        this.points = [];

        // Canvas elements
        this.imageCanvas = document.getElementById('imageCanvas');
        this.maskCanvas = document.getElementById('maskCanvas');
        this.imageCtx = this.imageCanvas.getContext('2d');
        this.maskCtx = this.maskCanvas.getContext('2d');

        // UI elements
        this.statusDiv = document.getElementById('status');
        this.statusText = document.getElementById('statusText');
        this.imageInput = document.getElementById('imageInput');
        this.resetBtn = document.getElementById('resetBtn');
        this.clearBtn = document.getElementById('clearBtn');
        this.showMaskToggle = document.getElementById('showMaskToggle');
        this.canvasContainer = document.getElementById('canvasContainer');
        this.pointsOverlay = document.getElementById('pointsOverlay');

        // Info elements
        this.pointCount = document.getElementById('pointCount');
        this.scoreValue = document.getElementById('scoreValue');
        this.imageSize = document.getElementById('imageSize');
        this.sessionIdEl = document.getElementById('sessionId');

        // Current image
        this.currentImage = null;

        this.init();
    }

    init() {
        this.connectWebSocket();
        this.setupEventListeners();
    }

    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/realtime`;

        this.updateStatus('connecting', 'Connecting to server...');

        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
            console.log('WebSocket connected');
        };

        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleServerMessage(data);
        };

        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.updateStatus('error', 'Connection error');
        };

        this.ws.onclose = () => {
            console.log('WebSocket disconnected');
            this.updateStatus('error', 'Disconnected from server');

            // Attempt to reconnect after 3 seconds
            setTimeout(() => {
                if (this.ws.readyState === WebSocket.CLOSED) {
                    this.connectWebSocket();
                }
            }, 3000);
        };
    }

    handleServerMessage(data) {
        console.log('Received:', data);

        switch (data.type) {
            case 'session_created':
                this.sessionId = data.session_id;
                this.sessionIdEl.textContent = data.session_id.substring(0, 8) + '...';
                this.updateStatus('connected', 'Connected - Ready to load image');
                break;

            case 'image_loaded':
                this.imageLoaded = true;
                this.imageSize.textContent = `${data.width} × ${data.height}`;
                this.updateStatus('connected', 'Image loaded - Click to segment');
                this.resetBtn.disabled = false;
                this.clearBtn.disabled = false;
                break;

            case 'point_added':
                this.points.push({ x: data.x, y: data.y, label: data.label });
                this.updatePointCount(data.total_points);
                this.renderPoints();
                break;

            case 'segmentation_result':
                this.displayMask(data.mask);
                this.scoreValue.textContent = (data.score * 100).toFixed(1) + '%';
                break;

            case 'points_cleared':
                this.points = [];
                this.updatePointCount(0);
                this.clearMask();
                this.clearPoints();
                this.scoreValue.textContent = '-';
                break;

            case 'error':
                console.error('Server error:', data.message);
                this.updateStatus('error', data.message);
                setTimeout(() => {
                    if (this.imageLoaded) {
                        this.updateStatus('connected', 'Ready to segment');
                    } else {
                        this.updateStatus('connected', 'Ready to load image');
                    }
                }, 3000);
                break;
        }
    }

    setupEventListeners() {
        // Image upload
        this.imageInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                this.loadImage(file);
            }
        });

        // Canvas click for segmentation
        this.imageCanvas.addEventListener('click', (e) => {
            if (!this.imageLoaded) return;
            this.handleCanvasClick(e, 1); // Foreground point
        });

        // Right-click for background points
        this.imageCanvas.addEventListener('contextmenu', (e) => {
            e.preventDefault();
            if (!this.imageLoaded) return;
            this.handleCanvasClick(e, 0); // Background point
        });

        // Reset button
        this.resetBtn.addEventListener('click', () => {
            this.resetPoints();
        });

        // Clear button
        this.clearBtn.addEventListener('click', () => {
            this.clearImage();
        });

        // Show mask toggle
        this.showMaskToggle.addEventListener('change', (e) => {
            this.maskCanvas.style.display = e.target.checked ? 'block' : 'none';
        });
    }

    loadImage(file) {
        const reader = new FileReader();

        reader.onload = (e) => {
            const img = new Image();

            img.onload = () => {
                this.currentImage = img;

                // Set canvas size to match image
                this.imageCanvas.width = img.width;
                this.imageCanvas.height = img.height;
                this.maskCanvas.width = img.width;
                this.maskCanvas.height = img.height;

                // Set canvas container size
                const containerAspect = this.canvasContainer.clientWidth / 600;
                const imageAspect = img.width / img.height;

                let displayWidth, displayHeight;
                if (imageAspect > containerAspect) {
                    displayWidth = this.canvasContainer.clientWidth;
                    displayHeight = displayWidth / imageAspect;
                } else {
                    displayHeight = 600;
                    displayWidth = displayHeight * imageAspect;
                }

                this.imageCanvas.style.width = displayWidth + 'px';
                this.imageCanvas.style.height = displayHeight + 'px';
                this.maskCanvas.style.width = displayWidth + 'px';
                this.maskCanvas.style.height = displayHeight + 'px';
                this.pointsOverlay.style.width = displayWidth + 'px';
                this.pointsOverlay.style.height = displayHeight + 'px';

                // Draw image
                this.imageCtx.drawImage(img, 0, 0);

                // Hide placeholder
                this.canvasContainer.querySelector('.placeholder').style.display = 'none';

                // Send to server
                const base64 = e.target.result;
                this.sendMessage({
                    type: 'init',
                    image: base64
                });

                this.updateStatus('connecting', 'Loading image...');
                this.resetPoints();
            };

            img.src = e.target.result;
        };

        reader.readAsDataURL(file);
    }

    handleCanvasClick(e, label) {
        const rect = this.imageCanvas.getBoundingClientRect();
        const scaleX = this.imageCanvas.width / rect.width;
        const scaleY = this.imageCanvas.height / rect.height;

        const x = Math.round((e.clientX - rect.left) * scaleX);
        const y = Math.round((e.clientY - rect.top) * scaleY);

        // Send click to server
        this.sendMessage({
            type: 'click',
            x: x,
            y: y,
            label: label
        });
    }

    displayMask(maskBase64) {
        if (!this.showMaskToggle.checked) return;

        const img = new Image();
        img.onload = () => {
            // Clear previous mask
            this.maskCtx.clearRect(0, 0, this.maskCanvas.width, this.maskCanvas.height);

            // Draw mask with transparency
            this.maskCtx.globalAlpha = 0.5;
            this.maskCtx.drawImage(img, 0, 0);
            this.maskCtx.globalAlpha = 1.0;
        };

        img.src = 'data:image/png;base64,' + maskBase64;
    }

    clearMask() {
        this.maskCtx.clearRect(0, 0, this.maskCanvas.width, this.maskCanvas.height);
    }

    renderPoints() {
        this.pointsOverlay.innerHTML = '';

        const rect = this.imageCanvas.getBoundingClientRect();
        const scaleX = rect.width / this.imageCanvas.width;
        const scaleY = rect.height / this.imageCanvas.height;

        this.points.forEach(point => {
            const marker = document.createElement('div');
            marker.className = `point-marker ${point.label === 1 ? 'foreground' : 'background'}`;
            marker.style.left = (point.x * scaleX) + 'px';
            marker.style.top = (point.y * scaleY) + 'px';
            this.pointsOverlay.appendChild(marker);
        });
    }

    clearPoints() {
        this.pointsOverlay.innerHTML = '';
    }

    resetPoints() {
        this.sendMessage({ type: 'reset' });
    }

    clearImage() {
        this.imageLoaded = false;
        this.currentImage = null;
        this.points = [];

        this.imageCtx.clearRect(0, 0, this.imageCanvas.width, this.imageCanvas.height);
        this.maskCtx.clearRect(0, 0, this.maskCanvas.width, this.maskCanvas.height);
        this.pointsOverlay.innerHTML = '';

        this.canvasContainer.querySelector('.placeholder').style.display = 'block';

        this.updatePointCount(0);
        this.scoreValue.textContent = '-';
        this.imageSize.textContent = '-';

        this.resetBtn.disabled = true;
        this.clearBtn.disabled = true;

        this.updateStatus('connected', 'Ready to load image');

        // Clear file input
        this.imageInput.value = '';
    }

    updatePointCount(count) {
        this.pointCount.textContent = count;
    }

    updateStatus(status, message) {
        this.statusDiv.className = `status ${status}`;
        this.statusText.textContent = message;
    }

    sendMessage(message) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(message));
        } else {
            console.error('WebSocket not connected');
        }
    }
}

// Initialize the app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.app = new RealtimeSAM();
});
